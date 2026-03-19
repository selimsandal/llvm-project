//===-- GPUSPIRVLowering.cpp - SPIR-V/OpenCL → GPU Lowering ===//
//
// IR-level pass that lowers SPIR-V and OpenCL builtins to GPU
// register reads, maps math builtins to LLVM intrinsics, strips
// OpenCL calling conventions and metadata.
//
// Handles two naming conventions:
//   1. SPIR-V: @_Z33__spirv_BuiltInGlobalInvocationIdi
//   2. OpenCL: @_Z13get_global_idj (C++ mangled)
//
// Current builtin mapping is based on raw workgroup state exposed through
// hidden launch context + I_GETSR:
//   group_id, local_size, num_groups, subgroup-local base, local_id
// Higher-level builtins like global_id are derived in the compiler from:
//   group_id * local_size + local_id
// Synchronization builtins lower through:
//   barrier(flags)    → llvm.gpu.workgroup.sync(mode)
//   mem_fence(flags)  → llvm.gpu.mem.fence(mode)
// Current verified source-level local-memory support on the optimized path:
//   __local kernel args → addrspace(3) pointer args treated as local byte offsets
//   atomic_add(__local*) → LLVM atomicrmw add → ATOMIC_LOCAL
// The current remaining gap is the unoptimized -O0 path, which still trips a
// backend FrameIndex selection failure for kernels that keep the wrapper allocas.
// Math builtins:
//   mad(a,b,c)        → llvm.fma.f32
//   min/max           → llvm.minnum/maxnum.f32
//
//===--------------------------------------------------------------===//

#include "GPU.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsGPU.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-spirv-lowering"

namespace {

class GPUSPIRVLowering : public ModulePass {
public:
  static char ID;
  GPUSPIRVLowering() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override {
    return "GPU SPIR-V Lowering";
  }

private:
  bool lowerBuiltinCalls(Module &M);
  bool lowerSyncBuiltins(Module &M);
  bool lowerAtomicBuiltins(Module &M);
  bool lowerMathBuiltins(Module &M);
  bool stripCallingConventions(Module &M);
  bool stripSPIRVMetadata(Module &M);
  bool removeDuplicateKernels(Module &M);
  bool promoteWrapperAllocas(Module &M);
};

char GPUSPIRVLowering::ID = 0;

} // anonymous namespace

enum OpenCLBuiltinKind {
  GPU_BUILTIN_NONE = 0,
  GPU_BUILTIN_GLOBAL_ID,
  GPU_BUILTIN_GROUP_ID,
  GPU_BUILTIN_LOCAL_ID,
  GPU_BUILTIN_GLOBAL_SIZE,
  GPU_BUILTIN_LOCAL_SIZE,
  GPU_BUILTIN_NUM_GROUPS,
  GPU_BUILTIN_BARRIER,
  GPU_BUILTIN_MEM_FENCE,
  GPU_BUILTIN_ATOMIC_ADD,
  GPU_BUILTIN_ATOMIC_XCHG,
  GPU_BUILTIN_ATOMIC_OR,
  GPU_BUILTIN_ATOMIC_AND,
  GPU_BUILTIN_ATOMIC_XOR,
  GPU_BUILTIN_ATOMIC_MIN,
  GPU_BUILTIN_ATOMIC_MAX,
  GPU_BUILTIN_ATOMIC_CMPXCHG,
};

enum GPUSReg : unsigned {
  SREG_GROUP_ID_X = 48,
  SREG_GROUP_ID_Y = 49,
  SREG_GROUP_ID_Z = 50,
  SREG_LOCAL_SIZE_X = 51,
  SREG_LOCAL_SIZE_Y = 52,
  SREG_LOCAL_SIZE_Z = 53,
  SREG_NUM_GROUPS_X = 54,
  SREG_NUM_GROUPS_Y = 55,
  SREG_NUM_GROUPS_Z = 56,
  SREG_LOCAL_ID_X = 59,
  SREG_LOCAL_ID_Y = 61,
  SREG_LOCAL_ID_Z = 62,
};

static Value *createGpuGetSR(IRBuilder<> &Builder, Module &M, unsigned SReg) {
  Function *GetSR = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::gpu_getsr,
                                                      {});
  return Builder.CreateCall(GetSR, {Builder.getInt32(SReg)});
}

static Value *createDimSelect(IRBuilder<> &Builder, Value *Dim,
                              Value *X, Value *Y, Value *Z) {
  Value *IsX = Builder.CreateICmpEQ(Dim, Builder.getInt32(0));
  Value *IsY = Builder.CreateICmpEQ(Dim, Builder.getInt32(1));
  return Builder.CreateSelect(IsX, X, Builder.CreateSelect(IsY, Y, Z));
}

static Value *createGroupID(IRBuilder<> &Builder, Module &M, Value *Dim) {
  Value *X = createGpuGetSR(Builder, M, SREG_GROUP_ID_X);
  Value *Y = createGpuGetSR(Builder, M, SREG_GROUP_ID_Y);
  Value *Z = createGpuGetSR(Builder, M, SREG_GROUP_ID_Z);
  return createDimSelect(Builder, Dim, X, Y, Z);
}

static Value *createLocalSize(IRBuilder<> &Builder, Module &M, Value *Dim) {
  Value *X = createGpuGetSR(Builder, M, SREG_LOCAL_SIZE_X);
  Value *Y = createGpuGetSR(Builder, M, SREG_LOCAL_SIZE_Y);
  Value *Z = createGpuGetSR(Builder, M, SREG_LOCAL_SIZE_Z);
  return createDimSelect(Builder, Dim, X, Y, Z);
}

static Value *createNumGroups(IRBuilder<> &Builder, Module &M, Value *Dim) {
  Value *X = createGpuGetSR(Builder, M, SREG_NUM_GROUPS_X);
  Value *Y = createGpuGetSR(Builder, M, SREG_NUM_GROUPS_Y);
  Value *Z = createGpuGetSR(Builder, M, SREG_NUM_GROUPS_Z);
  return createDimSelect(Builder, Dim, X, Y, Z);
}

static Value *createLocalID(IRBuilder<> &Builder, Module &M, Value *Dim) {
  Value *X = createGpuGetSR(Builder, M, SREG_LOCAL_ID_X);
  Value *Y = createGpuGetSR(Builder, M, SREG_LOCAL_ID_Y);
  Value *Z = createGpuGetSR(Builder, M, SREG_LOCAL_ID_Z);
  return createDimSelect(Builder, Dim, X, Y, Z);
}

static Value *createGlobalID(IRBuilder<> &Builder, Module &M, Value *Dim) {
  Value *GroupID = createGroupID(Builder, M, Dim);
  Value *LocalSize = createLocalSize(Builder, M, Dim);
  Value *LocalID = createLocalID(Builder, M, Dim);
  return Builder.CreateAdd(Builder.CreateMul(GroupID, LocalSize), LocalID,
                           "global_id");
}

static Value *createGlobalSize(IRBuilder<> &Builder, Module &M, Value *Dim) {
  Value *NumGroups = createNumGroups(Builder, M, Dim);
  Value *LocalSize = createLocalSize(Builder, M, Dim);
  return Builder.CreateMul(NumGroups, LocalSize, "global_size");
}

static unsigned mapOpenCLFenceFlagsToMode(uint32_t Flags) {
  constexpr uint32_t LocalBit = 1u;
  constexpr uint32_t GlobalBit = 2u;
  bool HasLocal = (Flags & LocalBit) != 0;
  bool HasGlobal = (Flags & GlobalBit) != 0;

  if (HasLocal && !HasGlobal)
    return 1;
  if (HasGlobal && !HasLocal)
    return 2;
  if (HasLocal || HasGlobal)
    return 3;
  return 0;
}

// Match SPIR-V and OpenCL builtin names for thread ID functions
static OpenCLBuiltinKind classifyBuiltin(StringRef Name) {
  // SPIR-V names (from SPIRV-LLVM-Translator metadata path)
  if (Name.contains("GlobalInvocationId")) return GPU_BUILTIN_GLOBAL_ID;
  if (Name.contains("WorkgroupId"))        return GPU_BUILTIN_GROUP_ID;
  if (Name.contains("LocalInvocationId"))  return GPU_BUILTIN_LOCAL_ID;
  if (Name.contains("GlobalSize"))         return GPU_BUILTIN_GLOBAL_SIZE;
  if (Name.contains("WorkgroupSize"))      return GPU_BUILTIN_LOCAL_SIZE;
  if (Name.contains("NumWorkgroups"))      return GPU_BUILTIN_NUM_GROUPS;

  // OpenCL C mangled names (from clang -target spir round-trip)
  // _Z13get_global_idj = get_global_id(uint)
  if (Name.contains("get_global_id"))      return GPU_BUILTIN_GLOBAL_ID;
  // _Z12get_group_idj = get_group_id(uint)
  if (Name.contains("get_group_id"))       return GPU_BUILTIN_GROUP_ID;
  // _Z12get_local_idj = get_local_id(uint)
  if (Name.contains("get_local_id"))       return GPU_BUILTIN_LOCAL_ID;
  if (Name.contains("get_global_size"))    return GPU_BUILTIN_GLOBAL_SIZE;
  if (Name.contains("get_local_size"))     return GPU_BUILTIN_LOCAL_SIZE;
  if (Name.contains("get_num_groups"))     return GPU_BUILTIN_NUM_GROUPS;
  if (Name.contains("barrier"))            return GPU_BUILTIN_BARRIER;
  if (Name.contains("mem_fence"))          return GPU_BUILTIN_MEM_FENCE;
  if (Name.contains("atomic_cmpxchg"))     return GPU_BUILTIN_ATOMIC_CMPXCHG;
  if (Name.contains("atomic_xchg"))        return GPU_BUILTIN_ATOMIC_XCHG;
  if (Name.contains("atomic_or"))          return GPU_BUILTIN_ATOMIC_OR;
  if (Name.contains("atomic_and"))         return GPU_BUILTIN_ATOMIC_AND;
  if (Name.contains("atomic_xor"))         return GPU_BUILTIN_ATOMIC_XOR;
  if (Name.contains("atomic_min"))         return GPU_BUILTIN_ATOMIC_MIN;
  if (Name.contains("atomic_max"))         return GPU_BUILTIN_ATOMIC_MAX;
  if (Name.contains("atomic_add"))         return GPU_BUILTIN_ATOMIC_ADD;

  return GPU_BUILTIN_NONE;
}

static bool isUnsignedAtomicMinMax(StringRef Name) {
  if (!Name.contains("atomic_min") && !Name.contains("atomic_max"))
    return false;
  return Name.contains("jj");
}

static bool isAtomicBuiltin(OpenCLBuiltinKind Kind) {
  switch (Kind) {
  case GPU_BUILTIN_ATOMIC_ADD:
  case GPU_BUILTIN_ATOMIC_XCHG:
  case GPU_BUILTIN_ATOMIC_OR:
  case GPU_BUILTIN_ATOMIC_AND:
  case GPU_BUILTIN_ATOMIC_XOR:
  case GPU_BUILTIN_ATOMIC_MIN:
  case GPU_BUILTIN_ATOMIC_MAX:
  case GPU_BUILTIN_ATOMIC_CMPXCHG:
    return true;
  default:
    return false;
  }
}

bool GPUSPIRVLowering::lowerBuiltinCalls(Module &M) {
  bool Changed = false;
  SmallVector<CallInst *, 16> ToReplace;
  SmallVector<Function *, 8> ToDelete;

  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;
    OpenCLBuiltinKind Kind = classifyBuiltin(F.getName());
    if (Kind == GPU_BUILTIN_NONE || Kind == GPU_BUILTIN_BARRIER ||
        Kind == GPU_BUILTIN_MEM_FENCE || isAtomicBuiltin(Kind))
      continue;

    for (User *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        ToReplace.push_back(CI);
    }

    for (CallInst *CI : ToReplace) {
      IRBuilder<> Builder(CI);
      Value *Result = nullptr;
      Value *Dim = CI->arg_size() ? CI->getArgOperand(0) : Builder.getInt32(0);

      switch (Kind) {
      case GPU_BUILTIN_GLOBAL_ID:
        Result = createGlobalID(Builder, M, Dim);
        break;
      case GPU_BUILTIN_GROUP_ID:
        Result = createGroupID(Builder, M, Dim);
        break;
      case GPU_BUILTIN_LOCAL_ID:
        Result = createLocalID(Builder, M, Dim);
        break;
      case GPU_BUILTIN_GLOBAL_SIZE:
        Result = createGlobalSize(Builder, M, Dim);
        break;
      case GPU_BUILTIN_LOCAL_SIZE:
        Result = createLocalSize(Builder, M, Dim);
        break;
      case GPU_BUILTIN_NUM_GROUPS:
        Result = createNumGroups(Builder, M, Dim);
        break;
      case GPU_BUILTIN_BARRIER:
      case GPU_BUILTIN_MEM_FENCE:
      case GPU_BUILTIN_ATOMIC_XCHG:
      case GPU_BUILTIN_ATOMIC_OR:
      case GPU_BUILTIN_ATOMIC_AND:
      case GPU_BUILTIN_ATOMIC_XOR:
      case GPU_BUILTIN_ATOMIC_MIN:
      case GPU_BUILTIN_ATOMIC_MAX:
      case GPU_BUILTIN_ATOMIC_CMPXCHG:
      case GPU_BUILTIN_ATOMIC_ADD:
      case GPU_BUILTIN_NONE:
        break;
      }

      if (Result) {
        CI->replaceAllUsesWith(Result);
        CI->eraseFromParent();
        Changed = true;
      }
    }
    ToReplace.clear();

    if (F.use_empty())
      ToDelete.push_back(&F);
  }

  for (Function *F : ToDelete)
    F->eraseFromParent();

  return Changed;
}

bool GPUSPIRVLowering::lowerSyncBuiltins(Module &M) {
  bool Changed = false;
  SmallVector<CallInst *, 8> ToReplace;
  SmallVector<Function *, 4> ToDelete;

  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;
    OpenCLBuiltinKind Kind = classifyBuiltin(F.getName());
    if (Kind != GPU_BUILTIN_BARRIER && Kind != GPU_BUILTIN_MEM_FENCE)
      continue;

    for (User *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        ToReplace.push_back(CI);
    }

    for (CallInst *CI : ToReplace) {
      IRBuilder<> Builder(CI);
      unsigned Mode = 0;
      if (CI->arg_size() != 0) {
        if (auto *Flags = dyn_cast<ConstantInt>(CI->getArgOperand(0)))
          Mode = mapOpenCLFenceFlagsToMode((uint32_t)Flags->getZExtValue());
      }

      Intrinsic::ID IntrID = (Kind == GPU_BUILTIN_BARRIER)
                                 ? Intrinsic::gpu_workgroup_sync
                                 : Intrinsic::gpu_mem_fence;
      Function *Sync = Intrinsic::getOrInsertDeclaration(&M, IntrID, {});
      Builder.CreateCall(Sync, {Builder.getInt32(Mode)});
      CI->eraseFromParent();
      Changed = true;
    }
    ToReplace.clear();

    if (F.use_empty())
      ToDelete.push_back(&F);
  }

  for (Function *F : ToDelete)
    F->eraseFromParent();

  return Changed;
}

bool GPUSPIRVLowering::lowerAtomicBuiltins(Module &M) {
  bool Changed = false;
  SmallVector<CallInst *, 8> ToReplace;
  SmallVector<Function *, 4> ToDelete;

  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;
    OpenCLBuiltinKind Kind = classifyBuiltin(F.getName());
    if (Kind != GPU_BUILTIN_ATOMIC_ADD &&
        Kind != GPU_BUILTIN_ATOMIC_XCHG &&
        Kind != GPU_BUILTIN_ATOMIC_OR &&
        Kind != GPU_BUILTIN_ATOMIC_AND &&
        Kind != GPU_BUILTIN_ATOMIC_XOR &&
        Kind != GPU_BUILTIN_ATOMIC_MIN &&
        Kind != GPU_BUILTIN_ATOMIC_MAX &&
        Kind != GPU_BUILTIN_ATOMIC_CMPXCHG)
      continue;

    for (User *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        ToReplace.push_back(CI);
    }

    for (CallInst *CI : ToReplace) {
      IRBuilder<> Builder(CI);
      Value *Result = nullptr;

      switch (Kind) {
      case GPU_BUILTIN_ATOMIC_ADD:
      case GPU_BUILTIN_ATOMIC_XCHG:
      case GPU_BUILTIN_ATOMIC_OR:
      case GPU_BUILTIN_ATOMIC_AND:
      case GPU_BUILTIN_ATOMIC_XOR:
      case GPU_BUILTIN_ATOMIC_MIN:
      case GPU_BUILTIN_ATOMIC_MAX: {
        if (CI->arg_size() < 2)
          break;
        Value *Ptr = CI->getArgOperand(0);
        Value *Val = CI->getArgOperand(1);
        AtomicRMWInst::BinOp Op = AtomicRMWInst::Add;
        switch (Kind) {
        case GPU_BUILTIN_ATOMIC_ADD:  Op = AtomicRMWInst::Add; break;
        case GPU_BUILTIN_ATOMIC_XCHG: Op = AtomicRMWInst::Xchg; break;
        case GPU_BUILTIN_ATOMIC_OR:   Op = AtomicRMWInst::Or; break;
        case GPU_BUILTIN_ATOMIC_AND:  Op = AtomicRMWInst::And; break;
        case GPU_BUILTIN_ATOMIC_XOR:  Op = AtomicRMWInst::Xor; break;
        case GPU_BUILTIN_ATOMIC_MIN:
          Op = isUnsignedAtomicMinMax(F.getName()) ? AtomicRMWInst::UMin
                                                   : AtomicRMWInst::Min;
          break;
        case GPU_BUILTIN_ATOMIC_MAX:
          Op = isUnsignedAtomicMinMax(F.getName()) ? AtomicRMWInst::UMax
                                                   : AtomicRMWInst::Max;
          break;
        case GPU_BUILTIN_ATOMIC_CMPXCHG:
        case GPU_BUILTIN_NONE:
        case GPU_BUILTIN_GLOBAL_ID:
        case GPU_BUILTIN_GROUP_ID:
        case GPU_BUILTIN_LOCAL_ID:
        case GPU_BUILTIN_GLOBAL_SIZE:
        case GPU_BUILTIN_LOCAL_SIZE:
        case GPU_BUILTIN_NUM_GROUPS:
        case GPU_BUILTIN_BARRIER:
        case GPU_BUILTIN_MEM_FENCE:
          break;
        }
        Result = Builder.CreateAtomicRMW(
            Op, Ptr, Val, MaybeAlign(4),
            AtomicOrdering::SequentiallyConsistent);
        break;
      }
      case GPU_BUILTIN_ATOMIC_CMPXCHG: {
        if (CI->arg_size() < 3)
          break;
        Value *Ptr = CI->getArgOperand(0);
        Value *Cmp = CI->getArgOperand(1);
        Value *Val = CI->getArgOperand(2);
        auto *Pair = Builder.CreateAtomicCmpXchg(
            Ptr, Cmp, Val, MaybeAlign(4), AtomicOrdering::SequentiallyConsistent,
            AtomicOrdering::SequentiallyConsistent);
        Result = Builder.CreateExtractValue(Pair, 0, "atomic_old");
        break;
      }
      case GPU_BUILTIN_NONE:
      case GPU_BUILTIN_GLOBAL_ID:
      case GPU_BUILTIN_GROUP_ID:
      case GPU_BUILTIN_LOCAL_ID:
      case GPU_BUILTIN_GLOBAL_SIZE:
      case GPU_BUILTIN_LOCAL_SIZE:
      case GPU_BUILTIN_NUM_GROUPS:
      case GPU_BUILTIN_BARRIER:
      case GPU_BUILTIN_MEM_FENCE:
        break;
      }

      if (Result) {
        CI->replaceAllUsesWith(Result);
        CI->eraseFromParent();
        Changed = true;
      }
    }
    ToReplace.clear();

    if (F.use_empty())
      ToDelete.push_back(&F);
  }

  for (Function *F : ToDelete)
    F->eraseFromParent();

  return Changed;
}

static bool promoteSimpleAllocas(Function &F) {
  if (F.isDeclaration() || F.empty())
    return false;

  BasicBlock &Entry = F.getEntryBlock();
  SmallVector<AllocaInst *, 8> Allocas;
  for (Instruction &I : Entry) {
    auto *AI = dyn_cast<AllocaInst>(&I);
    if (!AI)
      continue;
    if (isAllocaPromotable(AI))
      Allocas.push_back(AI);
  }

  if (Allocas.empty())
    return false;

  DominatorTree DT(F);
  PromoteMemToReg(Allocas, DT);
  return true;
}

bool GPUSPIRVLowering::promoteWrapperAllocas(Module &M) {
  bool Changed = false;
  for (Function &F : M)
    Changed |= promoteSimpleAllocas(F);
  return Changed;
}

// Replace OpenCL math builtins with LLVM intrinsics
bool GPUSPIRVLowering::lowerMathBuiltins(Module &M) {
  bool Changed = false;
  SmallVector<CallInst *, 16> ToReplace;
  SmallVector<Function *, 8> ToDelete;

  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;
    StringRef Name = F.getName();

    // Classify math builtins by name prefix.
    // OpenCL C++ mangles as: _Z3madfff, _Z3minii, _Z3maxff, etc.
    // The suffix encodes argument types (f=float, i=int, j=uint).
    enum { NONE, MAD, MIN, MAX } Kind = NONE;
    if (Name.starts_with("_Z3mad") || Name.starts_with("_Z4fmad"))
      Kind = MAD;
    else if (Name.starts_with("_Z3min") || Name.starts_with("_Z4fmin"))
      Kind = MIN;
    else if (Name.starts_with("_Z3max") || Name.starts_with("_Z4fmax"))
      Kind = MAX;

    if (Kind == NONE)
      continue;

    for (User *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        ToReplace.push_back(CI);
    }

    for (CallInst *CI : ToReplace) {
      IRBuilder<> Builder(CI);
      Value *Result = nullptr;
      bool IsFloat = CI->getArgOperand(0)->getType()->isFloatTy();

      if (Kind == MAD && CI->arg_size() == 3) {
        Function *FMA = Intrinsic::getOrInsertDeclaration(
            &M, Intrinsic::fma, {Builder.getFloatTy()});
        Result = Builder.CreateCall(FMA,
            {CI->getArgOperand(0), CI->getArgOperand(1),
             CI->getArgOperand(2)});
      } else if (Kind == MIN && CI->arg_size() == 2) {
        if (IsFloat) {
          Function *Fn = Intrinsic::getOrInsertDeclaration(
              &M, Intrinsic::minnum, {Builder.getFloatTy()});
          Result = Builder.CreateCall(Fn,
              {CI->getArgOperand(0), CI->getArgOperand(1)});
        } else {
          // _Z3minii → smin, _Z3minjj → umin
          bool IsUnsigned = Name.contains('j');
          Function *Fn = Intrinsic::getOrInsertDeclaration(
              &M, IsUnsigned ? Intrinsic::umin : Intrinsic::smin,
              {CI->getArgOperand(0)->getType()});
          Result = Builder.CreateCall(Fn,
              {CI->getArgOperand(0), CI->getArgOperand(1)});
        }
      } else if (Kind == MAX && CI->arg_size() == 2) {
        if (IsFloat) {
          Function *Fn = Intrinsic::getOrInsertDeclaration(
              &M, Intrinsic::maxnum, {Builder.getFloatTy()});
          Result = Builder.CreateCall(Fn,
              {CI->getArgOperand(0), CI->getArgOperand(1)});
        } else {
          bool IsUnsigned = Name.contains('j');
          Function *Fn = Intrinsic::getOrInsertDeclaration(
              &M, IsUnsigned ? Intrinsic::umax : Intrinsic::smax,
              {CI->getArgOperand(0)->getType()});
          Result = Builder.CreateCall(Fn,
              {CI->getArgOperand(0), CI->getArgOperand(1)});
        }
      }

      if (Result) {
        CI->replaceAllUsesWith(Result);
        CI->eraseFromParent();
        Changed = true;
      }
    }
    ToReplace.clear();

    if (F.use_empty())
      ToDelete.push_back(&F);
  }

  for (Function *F : ToDelete)
    F->eraseFromParent();

  return Changed;
}

// Strip spir_kernel / spir_func calling conventions → default
bool GPUSPIRVLowering::stripCallingConventions(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL ||
        F.getCallingConv() == CallingConv::SPIR_FUNC) {
      F.setCallingConv(CallingConv::C);
      Changed = true;
    }
  }
  return Changed;
}

// Remove SPIR-V and OpenCL metadata
bool GPUSPIRVLowering::stripSPIRVMetadata(Module &M) {
  bool Changed = false;
  SmallVector<StringRef, 8> ToErase;

  for (auto &NMD : M.named_metadata()) {
    StringRef Name = NMD.getName();
    if (Name.starts_with("spirv.") || Name.starts_with("opencl."))
      ToErase.push_back(Name);
  }

  for (StringRef Name : ToErase) {
    if (auto *NMD = M.getNamedMetadata(Name)) {
      NMD->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

// Remove __clang_ocl_kern_imp_* duplicate functions
bool GPUSPIRVLowering::removeDuplicateKernels(Module &M) {
  bool Changed = false;
  SmallVector<Function *, 4> ToDelete;
  SmallVector<ReturnInst *, 4> Returns;
  constexpr StringLiteral ImplPrefix = "__clang_ocl_kern_imp_";

  for (Function &F : M) {
    if (!F.getName().starts_with(ImplPrefix))
      continue;

    StringRef WrapperName = F.getName().drop_front(ImplPrefix.size());
    Function *Wrapper = M.getFunction(WrapperName);
    if (!Wrapper || Wrapper == &F)
      continue;
    if (Wrapper->arg_size() != F.arg_size())
      continue;

    ValueToValueMapTy VMap;
    auto WI = Wrapper->arg_begin();
    for (Argument &Arg : F.args())
      VMap[&Arg] = &*WI++;

    Wrapper->deleteBody();
    Returns.clear();
    CloneFunctionInto(Wrapper, &F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                      Returns);
    ToDelete.push_back(&F);
    Changed = true;
  }

  for (Function *F : ToDelete) {
    if (F->use_empty())
      F->eraseFromParent();
  }

  return Changed;
}

bool GPUSPIRVLowering::runOnModule(Module &M) {
  bool Changed = false;
  Changed |= lowerBuiltinCalls(M);
  Changed |= lowerSyncBuiltins(M);
  Changed |= lowerAtomicBuiltins(M);
  Changed |= lowerMathBuiltins(M);
  Changed |= stripCallingConventions(M);
  Changed |= stripSPIRVMetadata(M);
  Changed |= removeDuplicateKernels(M);
  Changed |= promoteWrapperAllocas(M);
  return Changed;
}

INITIALIZE_PASS(GPUSPIRVLowering, "gpu-spirv-lowering",
                "GPU SPIR-V Lowering", false, false)

namespace llvm {
ModulePass *createGPUSPIRVLoweringPass() {
  return new GPUSPIRVLowering();
}
} // namespace llvm
