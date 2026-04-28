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
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsGPU.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
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
  bool lowerResourceAccess(Module &M);
  bool lowerMathBuiltins(Module &M);
  bool stripCallingConventions(Module &M);
  bool stripSPIRVMetadata(Module &M);
  bool removeDuplicateKernels(Module &M);
  bool promoteWrapperAllocas(Module &M);
  bool markHelpersAlwaysInline(Module &M);
  bool inlineAlwaysInlineHelpers(Module &M);
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
  GPU_BUILTIN_FLATTENED_LOCAL_ID,
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

static Value *createRegRead(IRBuilder<> &Builder, Module &M,
                            const char *RegName) {
  Function *ReadReg = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::read_register, {Builder.getInt32Ty()});
  MDNode *RegMD =
      MDNode::get(M.getContext(), {MDString::get(M.getContext(), RegName)});
  return Builder.CreateCall(
      ReadReg, {MetadataAsValue::get(M.getContext(), RegMD)});
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

static Value *createFlattenedLocalID(IRBuilder<> &Builder, Module &M) {
  Value *X = createGpuGetSR(Builder, M, SREG_LOCAL_ID_X);
  Value *Y = createGpuGetSR(Builder, M, SREG_LOCAL_ID_Y);
  Value *Z = createGpuGetSR(Builder, M, SREG_LOCAL_ID_Z);
  Value *SizeX = createGpuGetSR(Builder, M, SREG_LOCAL_SIZE_X);
  Value *SizeY = createGpuGetSR(Builder, M, SREG_LOCAL_SIZE_Y);
  Value *ScaledZ = Builder.CreateMul(Z, SizeY, "lin_z");
  Value *YZ = Builder.CreateAdd(Y, ScaledZ, "lin_yz");
  return Builder.CreateAdd(X, Builder.CreateMul(YZ, SizeX, "lin_xy"),
                           "lin_idx");
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
  // DirectX/HLSL intrinsics are handled by GPUHLSLLowering. Keep this pass from
  // matching broad words like "barrier" inside llvm.dx.* names before that pass
  // gets a chance to apply HLSL-specific semantics.
  if (Name.starts_with("llvm.dx."))
    return GPU_BUILTIN_NONE;

  // SPIR-V names (from SPIRV-LLVM-Translator metadata path)
  if (Name.contains("GlobalInvocationId")) return GPU_BUILTIN_GLOBAL_ID;
  if (Name.contains("WorkgroupId"))        return GPU_BUILTIN_GROUP_ID;
  if (Name.contains("LocalInvocationId"))  return GPU_BUILTIN_LOCAL_ID;
  if (Name.contains("GlobalSize"))         return GPU_BUILTIN_GLOBAL_SIZE;
  if (Name.contains("WorkgroupSize"))      return GPU_BUILTIN_LOCAL_SIZE;
  if (Name.contains("NumWorkgroups"))      return GPU_BUILTIN_NUM_GROUPS;

  // HLSL SPIR-V frontend intrinsics.
  if (Name.starts_with("llvm.spv.thread.id.in.group"))
    return GPU_BUILTIN_LOCAL_ID;
  if (Name.starts_with("llvm.spv.thread.id"))
    return GPU_BUILTIN_GLOBAL_ID;
  if (Name.starts_with("llvm.spv.group.id"))
    return GPU_BUILTIN_GROUP_ID;
  if (Name.starts_with("llvm.spv.flattened.thread.id.in.group"))
    return GPU_BUILTIN_FLATTENED_LOCAL_ID;

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

enum SPVResourceKind {
  SPV_RESOURCE_NONE = 0,
  SPV_HANDLE_FROM_BINDING,
  SPV_COUNTER_HANDLE_FROM_IMPLICIT,
  SPV_RESOURCE_GETPOINTER,
};

static SPVResourceKind classifySPVResource(StringRef Name) {
  if (Name.starts_with("llvm.spv.resource.handlefrombinding"))
    return SPV_HANDLE_FROM_BINDING;
  if (Name.starts_with("llvm.spv.resource.counterhandlefromimplicitbinding"))
    return SPV_COUNTER_HANDLE_FROM_IMPLICIT;
  if (Name.starts_with("llvm.spv.resource.getpointer"))
    return SPV_RESOURCE_GETPOINTER;
  return SPV_RESOURCE_NONE;
}

static bool isSPVVulkanBufferHandleType(Type *Ty) {
  auto *TET = dyn_cast<TargetExtType>(Ty);
  return TET && TET->getName() == "spirv.VulkanBuffer";
}

static bool typeContainsSPVVulkanBuffer(Type *Ty) {
  if (isSPVVulkanBufferHandleType(Ty))
    return true;
  if (auto *ST = dyn_cast<StructType>(Ty)) {
    for (Type *ElemTy : ST->elements())
      if (typeContainsSPVVulkanBuffer(ElemTy))
        return true;
  }
  if (auto *AT = dyn_cast<ArrayType>(Ty))
    return typeContainsSPVVulkanBuffer(AT->getElementType());
  return false;
}

static Value *getSPVHandleStorageKey(Value *Ptr) {
  Ptr = Ptr->stripPointerCasts();

  auto *GEP = dyn_cast<GEPOperator>(Ptr);
  if (!GEP)
    return Ptr;

  for (Value *Idx : GEP->indices()) {
    auto *CI = dyn_cast<ConstantInt>(Idx);
    if (!CI || !CI->isZero())
      return Ptr;
  }

  return getSPVHandleStorageKey(GEP->getPointerOperand());
}

static bool getStoredConstantUInt(Value *Ptr, unsigned &Out) {
  Value *Key = getSPVHandleStorageKey(Ptr);
  ConstantInt *Stored = nullptr;

  for (User *U : Key->users()) {
    auto *SI = dyn_cast<StoreInst>(U);
    if (!SI || getSPVHandleStorageKey(SI->getPointerOperand()) != Key)
      continue;
    auto *CI = dyn_cast<ConstantInt>(SI->getValueOperand());
    if (!CI)
      return false;
    if (Stored && Stored->getZExtValue() != CI->getZExtValue())
      return false;
    Stored = CI;
  }

  if (!Stored)
    return false;
  Out = Stored->getZExtValue();
  return true;
}

static bool getConstantUInt(Value *V, unsigned &Out) {
  if (auto *CI = dyn_cast<ConstantInt>(V)) {
    Out = CI->getZExtValue();
    return true;
  }
  if (auto *LI = dyn_cast<LoadInst>(V))
    return getStoredConstantUInt(LI->getPointerOperand(), Out);
  return false;
}

struct SPVResourceHandleInfo {
  unsigned BindSlot = 0;
  Type *HandleType = nullptr;
};

static bool getDirectSPVHandleInfo(Value *Handle,
                                   SPVResourceHandleInfo &Info) {
  auto *CI = dyn_cast<CallInst>(Handle);
  if (!CI || !CI->getCalledFunction())
    return false;
  if (!isSPVVulkanBufferHandleType(Handle->getType()))
    return false;

  if (classifySPVResource(CI->getCalledFunction()->getName()) !=
      SPV_HANDLE_FROM_BINDING)
    return false;

  // handlefrombinding args: (space, binding, range_size, index, name_ptr)
  unsigned BindSlot = 0;
  if (!getConstantUInt(CI->getArgOperand(1), BindSlot))
    return false;

  Info = {BindSlot, Handle->getType()};
  return true;
}

static bool getSPVHandleInfo(
    Value *Handle, const DenseMap<Value *, SPVResourceHandleInfo> &StoredHandles,
    SPVResourceHandleInfo &Info) {
  if (getDirectSPVHandleInfo(Handle, Info))
    return true;

  auto *LI = dyn_cast<LoadInst>(Handle);
  if (!LI || !isSPVVulkanBufferHandleType(LI->getType()))
    return false;

  auto It = StoredHandles.find(getSPVHandleStorageKey(LI->getPointerOperand()));
  if (It == StoredHandles.end())
    return false;

  Info = It->second;
  return true;
}

static const char *bindSlotToReg(unsigned Slot) {
  switch (Slot) {
  case 0:
    return "r1";
  case 1:
    return "r2";
  case 2:
    return "r3";
  case 3:
    return "r4";
  default:
    return nullptr;
  }
}

static Value *getBindingBase(IRBuilder<> &Builder, Module &M,
                             unsigned BindSlot, bool UseIndirect) {
  if (!UseIndirect) {
    const char *RegName = bindSlotToReg(BindSlot);
    if (!RegName)
      return nullptr;
    return createRegRead(Builder, M, RegName);
  }

  Value *ArgsPtr = createRegRead(Builder, M, "r1");
  Value *Offset = Builder.getInt32(BindSlot * 4);
  Value *Addr = Builder.CreateAdd(ArgsPtr, Offset, "arg_addr");
  Value *Ptr = Builder.CreateIntToPtr(
      Addr, PointerType::getUnqual(M.getContext()), "arg_ptr");
  return Builder.CreateLoad(Builder.getInt32Ty(), Ptr, "arg_val");
}

static unsigned getSPVResourceElementSize(Type *HandleType,
                                          const DataLayout &DL) {
  auto *TET = dyn_cast<TargetExtType>(HandleType);
  if (!TET || !isSPVVulkanBufferHandleType(HandleType) ||
      TET->getNumTypeParameters() == 0)
    return 4;

  Type *ElemTy = TET->getTypeParameter(0);
  if (auto *ArrayTy = dyn_cast<ArrayType>(ElemTy))
    ElemTy = ArrayTy->getElementType();

  return DL.getTypeAllocSize(ElemTy);
}

static Value *createResourceAddress(IRBuilder<> &Builder, Value *Base,
                                    Value *Index, unsigned ElemSize) {
  if (ElemSize == 1)
    return Builder.CreateAdd(Base, Index, "addr");

  Value *Offset =
      Builder.CreateMul(Index, Builder.getInt32(ElemSize), "idx_offset");
  return Builder.CreateAdd(Base, Offset, "addr");
}

static bool classifySPVSyncIntrinsic(StringRef Name, Intrinsic::ID &IntrID,
                                     unsigned &Mode) {
  if (Name == "llvm.spv.group.memory.barrier") {
    IntrID = Intrinsic::gpu_mem_fence;
    Mode = 1;
    return true;
  }
  if (Name == "llvm.spv.device.memory.barrier") {
    IntrID = Intrinsic::gpu_mem_fence;
    Mode = 2;
    return true;
  }
  if (Name == "llvm.spv.all.memory.barrier") {
    IntrID = Intrinsic::gpu_mem_fence;
    Mode = 3;
    return true;
  }
  if (Name == "llvm.spv.group.memory.barrier.with.group.sync") {
    IntrID = Intrinsic::gpu_workgroup_sync;
    Mode = 1;
    return true;
  }
  if (Name == "llvm.spv.device.memory.barrier.with.group.sync") {
    IntrID = Intrinsic::gpu_workgroup_sync;
    Mode = 2;
    return true;
  }
  if (Name == "llvm.spv.all.memory.barrier.with.group.sync") {
    IntrID = Intrinsic::gpu_workgroup_sync;
    Mode = 3;
    return true;
  }
  return false;
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
      case GPU_BUILTIN_FLATTENED_LOCAL_ID:
        Result = createFlattenedLocalID(Builder, M);
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
    Intrinsic::ID FixedIntrID = Intrinsic::not_intrinsic;
    unsigned FixedMode = 0;
    bool IsFixedSPVSync =
        classifySPVSyncIntrinsic(F.getName(), FixedIntrID, FixedMode);
    OpenCLBuiltinKind Kind = IsFixedSPVSync ? GPU_BUILTIN_NONE
                                            : classifyBuiltin(F.getName());
    if (!IsFixedSPVSync &&
        Kind != GPU_BUILTIN_BARRIER && Kind != GPU_BUILTIN_MEM_FENCE)
      continue;

    for (User *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        ToReplace.push_back(CI);
    }

    for (CallInst *CI : ToReplace) {
      IRBuilder<> Builder(CI);
      unsigned Mode = FixedMode;
      if (!IsFixedSPVSync && CI->arg_size() != 0) {
        if (auto *Flags = dyn_cast<ConstantInt>(CI->getArgOperand(0)))
          Mode = mapOpenCLFenceFlagsToMode((uint32_t)Flags->getZExtValue());
      }

      Intrinsic::ID IntrID =
          IsFixedSPVSync ? FixedIntrID
                         : ((Kind == GPU_BUILTIN_BARRIER)
                                ? Intrinsic::gpu_workgroup_sync
                                : Intrinsic::gpu_mem_fence);
      Function *Sync = Intrinsic::getOrInsertDeclaration(&M, IntrID, {});
      SmallVector<OperandBundleDef, 1> Bundles;
      CI->getOperandBundlesAsDefs(Bundles);
      Builder.CreateCall(Sync, {Builder.getInt32(Mode)}, Bundles);
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
        case GPU_BUILTIN_FLATTENED_LOCAL_ID:
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
      case GPU_BUILTIN_FLATTENED_LOCAL_ID:
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

bool GPUSPIRVLowering::lowerResourceAccess(Module &M) {
  bool Changed = false;
  const DataLayout &DL = M.getDataLayout();
  SmallVector<CallInst *, 16> GetPtrCalls;
  SmallVector<CallInst *, 16> HandleCalls;
  DenseMap<Value *, SPVResourceHandleInfo> StoredHandles;

  for (Function &F : M) {
    SPVResourceKind Kind = classifySPVResource(F.getName());
    if (Kind == SPV_RESOURCE_NONE)
      continue;

    for (User *U : F.users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;
      if (Kind == SPV_RESOURCE_GETPOINTER)
        GetPtrCalls.push_back(CI);
      else
        HandleCalls.push_back(CI);
    }
  }

  bool LocalChanged = true;
  while (LocalChanged) {
    LocalChanged = false;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          auto *SI = dyn_cast<StoreInst>(&I);
          if (!SI || !isSPVVulkanBufferHandleType(
                         SI->getValueOperand()->getType()))
            continue;

          SPVResourceHandleInfo Info;
          if (!getSPVHandleInfo(SI->getValueOperand(), StoredHandles, Info))
            continue;

          Value *Key = getSPVHandleStorageKey(SI->getPointerOperand());
          auto It = StoredHandles.find(Key);
          if (It != StoredHandles.end() &&
              It->second.BindSlot == Info.BindSlot &&
              It->second.HandleType == Info.HandleType)
            continue;

          StoredHandles[Key] = Info;
          LocalChanged = true;
        }
      }
    }
  }

  unsigned MaxBindSlot = 0;
  for (CallInst *CI : HandleCalls) {
    SPVResourceHandleInfo Info;
    if (getDirectSPVHandleInfo(CI, Info))
      MaxBindSlot = std::max(MaxBindSlot, Info.BindSlot);
  }
  for (const auto &Entry : StoredHandles) {
    MaxBindSlot = std::max(MaxBindSlot, Entry.second.BindSlot);
  }
  bool UseIndirect = MaxBindSlot > 3;
  bool AllBufferGetPointersLowered = true;

  for (CallInst *CI : GetPtrCalls) {
    if (CI->arg_size() < 2)
      continue;

    Value *Handle = CI->getArgOperand(0);
    if (!isSPVVulkanBufferHandleType(Handle->getType()))
      continue;

    SPVResourceHandleInfo Info;
    if (!getSPVHandleInfo(Handle, StoredHandles, Info)) {
      AllBufferGetPointersLowered = false;
      continue;
    }

    IRBuilder<> Builder(CI);
    Value *Base = getBindingBase(Builder, M, Info.BindSlot, UseIndirect);
    if (!Base)
      continue;

    unsigned ElemSize = getSPVResourceElementSize(Info.HandleType, DL);
    Value *Addr =
        createResourceAddress(Builder, Base, CI->getArgOperand(1), ElemSize);
    Value *Ptr = Builder.CreateIntToPtr(Addr, CI->getType(), "buf_ptr");

    CI->replaceAllUsesWith(Ptr);
    CI->eraseFromParent();
    Changed = true;
  }

  if (AllBufferGetPointersLowered) {
    SmallVector<Instruction *, 32> DeadHandleInsts;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto *LI = dyn_cast<LoadInst>(&I)) {
            if (isSPVVulkanBufferHandleType(LI->getType())) {
              if (!LI->use_empty())
                LI->replaceAllUsesWith(PoisonValue::get(LI->getType()));
              DeadHandleInsts.push_back(LI);
            }
            continue;
          }

          auto *SI = dyn_cast<StoreInst>(&I);
          if (SI && isSPVVulkanBufferHandleType(
                        SI->getValueOperand()->getType()))
            DeadHandleInsts.push_back(SI);
        }
      }
    }

    for (Instruction *I : DeadHandleInsts) {
      I->eraseFromParent();
      Changed = true;
    }

    bool RemovedScaffolding = true;
    while (RemovedScaffolding) {
      RemovedScaffolding = false;
      SmallVector<Instruction *, 16> DeadScaffolding;
      for (Function &F : M) {
        if (F.isDeclaration())
          continue;

        for (BasicBlock &BB : F) {
          for (Instruction &I : BB) {
            if (!I.use_empty())
              continue;

            if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
              if (typeContainsSPVVulkanBuffer(GEP->getSourceElementType()))
                DeadScaffolding.push_back(GEP);
              continue;
            }

            auto *AI = dyn_cast<AllocaInst>(&I);
            if (AI && typeContainsSPVVulkanBuffer(AI->getAllocatedType()))
              DeadScaffolding.push_back(AI);
          }
        }
      }

      for (Instruction *I : DeadScaffolding) {
        I->eraseFromParent();
        Changed = true;
        RemovedScaffolding = true;
      }
    }
  }

  for (CallInst *CI : HandleCalls) {
    if (CI->use_empty()) {
      CI->eraseFromParent();
      Changed = true;
    }
  }

  SmallVector<Function *, 8> DeadFuncs;
  for (Function &F : M) {
    if (classifySPVResource(F.getName()) != SPV_RESOURCE_NONE && F.use_empty())
      DeadFuncs.push_back(&F);
  }
  for (Function *F : DeadFuncs)
    F->eraseFromParent();

  SmallVector<GlobalVariable *, 8> DeadGlobals;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.use_empty() && GV.hasLocalLinkage() &&
        typeContainsSPVVulkanBuffer(GV.getValueType()))
      DeadGlobals.push_back(&GV);
  }
  for (GlobalVariable *GV : DeadGlobals) {
    GV->eraseFromParent();
    Changed = true;
  }

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
    enum { NONE, MAD, MIN, MAX, FABS_KIND, MUL24_KIND, MAD24_KIND }
        Kind = NONE;
    if (Name.starts_with("_Z3mad") || Name.starts_with("_Z4fmad") ||
        Name.starts_with("_Z3fma"))
      Kind = MAD;
    else if (Name.starts_with("_Z3min") || Name.starts_with("_Z4fmin"))
      Kind = MIN;
    else if (Name.starts_with("_Z3max") || Name.starts_with("_Z4fmax"))
      Kind = MAX;
    else if (Name.starts_with("_Z4fabs"))
      Kind = FABS_KIND;
    else if (Name.starts_with("_Z5mul24"))
      Kind = MUL24_KIND;
    else if (Name.starts_with("_Z5mad24"))
      Kind = MAD24_KIND;

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
      } else if (Kind == FABS_KIND && CI->arg_size() == 1 && IsFloat) {
        // OpenCL `fabs(float)` → llvm.fabs.f32. The GPU backend has
        // ISD::FABS set to Expand, which the LegalizeDAG pass turns into
        // `and x, 0x7fffffff`, which GPUPeephole then folds into a real
        // FABS source modifier. Without this rewrite the call remained as
        // a `_Z4fabsf` libcall and crashed call lowering.
        Function *Fn = Intrinsic::getOrInsertDeclaration(
            &M, Intrinsic::fabs, {Builder.getFloatTy()});
        Result = Builder.CreateCall(Fn, {CI->getArgOperand(0)});
      } else if (Kind == MUL24_KIND && CI->arg_size() == 2) {
        // OpenCL `mul24(a, b)` promises only the low 32 bits of the
        // product of the low 24 bits of each operand. The GPU's `IMUL`
        // is already a full 32x32→32 multiply, so a plain `mul` gives
        // the same low 32 bits. Clang emits this as the libcall
        // `_Z5mul24ii` / `_Z5mul24jj`, which has no backend handler and
        // crashes call lowering; rewriting to `mul` skips that path.
        Result = Builder.CreateMul(CI->getArgOperand(0),
                                   CI->getArgOperand(1));
      } else if (Kind == MAD24_KIND && CI->arg_size() == 3) {
        // OpenCL `mad24(a, b, c)` = `mul24(a, b) + c`; same reasoning as
        // above — plain mul+add gives the specified low-bit result.
        Value *M = Builder.CreateMul(CI->getArgOperand(0),
                                     CI->getArgOperand(1));
        Result = Builder.CreateAdd(M, CI->getArgOperand(2));
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

// Returns true if the type is one the GPU calling convention can carry
// in r1..r4 / r0 (32-bit scalar integer, 32-bit float, or pointer).
// Anything else — vectors, aggregates, larger integers — has no
// argument/return slot in this backend, so a function whose signature
// uses such a type can never be an entry point.
static bool isCallingConvCompatibleType(Type *T) {
  if (T->isVoidTy())
    return true;
  if (T->isFloatTy())
    return true;
  if (T->isPointerTy())
    return true;
  if (auto *IT = dyn_cast<IntegerType>(T))
    return IT->getBitWidth() <= 32;
  return false;
}

static bool hasCallingConvCompatibleSignature(const Function &F) {
  if (!isCallingConvCompatibleType(F.getReturnType()))
    return false;
  for (const Argument &A : F.args())
    if (!isCallingConvCompatibleType(A.getType()))
      return false;
  return true;
}

// Mark non-kernel HELPER functions `alwaysinline` and give them
// internal linkage so the later AlwaysInlinerLegacyPass + GlobalDCE
// fold helper functions completely away. The GPU backend only models
// scalar i32/f32 and has no calling-convention slot for vector or
// struct return values, so helpers that return `float4` or similar
// must disappear before ISel sees them; this is what unblocks
// Rodinia's hybridsort/mergesort where `sortElem` returns float4.
//
// We have to be careful with non-kernel functions that might be
// real entry points: lit-style `define void @test_add(...)` tests
// have no callers, and inlining them away would leave the module
// empty. Distinguish helpers from entry points by checking the
// signature — if a function uses a type the GPU calling convention
// cannot pass (vector/aggregate/i64/...) it CAN'T be an entry point,
// so internalize it even when use_empty(); otherwise leave
// callerless functions alone so the lit tests survive.
bool GPUSPIRVLowering::markHelpersAlwaysInline(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
      continue;
    // `optnone` is a deliberate `-O0` choice (the IR verifier even
    // requires it to come paired with `noinline`); leave those alone.
    if (F.hasFnAttribute(Attribute::OptimizeNone))
      continue;

    bool CcCompatible = hasCallingConvCompatibleSignature(F);
    if (CcCompatible && F.use_empty())
      continue; // looks like a top-level entry point — leave it alone.

    if (F.hasFnAttribute(Attribute::NoInline)) {
      F.removeFnAttr(Attribute::NoInline);
      Changed = true;
    }
    if (!F.hasFnAttribute(Attribute::AlwaysInline)) {
      F.addFnAttr(Attribute::AlwaysInline);
      Changed = true;
    }
    if (!F.hasLocalLinkage()) {
      F.setLinkage(GlobalValue::InternalLinkage);
      Changed = true;
    }
  }
  return Changed;
}

bool GPUSPIRVLowering::inlineAlwaysInlineHelpers(Module &M) {
  bool Changed = false;
  bool LocalChanged = true;

  while (LocalChanged) {
    LocalChanged = false;
    SmallVector<CallBase *, 16> Calls;

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          auto *CB = dyn_cast<CallBase>(&I);
          if (!CB)
            continue;

          Function *Callee = CB->getCalledFunction();
          if (!Callee || Callee->isDeclaration() || Callee == &F)
            continue;
          if (!Callee->hasFnAttribute(Attribute::AlwaysInline))
            continue;

          Calls.push_back(CB);
        }
      }
    }

    for (CallBase *CB : Calls) {
      if (!CB->getParent())
        continue;
      Function *Callee = CB->getCalledFunction();
      if (!Callee || Callee->isDeclaration() ||
          !Callee->hasFnAttribute(Attribute::AlwaysInline))
        continue;

      InlineFunctionInfo IFI;
      InlineResult Res = InlineFunction(*CB, IFI);
      if (!Res.isSuccess())
        continue;

      Changed = true;
      LocalChanged = true;
    }
  }

  SmallVector<Function *, 16> DeadHelpers;
  for (Function &F : M) {
    if (!F.isDeclaration() && F.hasLocalLinkage() && F.use_empty() &&
        F.hasFnAttribute(Attribute::AlwaysInline) &&
        F.getCallingConv() != CallingConv::SPIR_KERNEL)
      DeadHelpers.push_back(&F);
  }

  for (Function *F : DeadHelpers) {
    F->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

bool GPUSPIRVLowering::runOnModule(Module &M) {
  bool Changed = false;
  Changed |= lowerBuiltinCalls(M);
  Changed |= lowerSyncBuiltins(M);
  Changed |= lowerAtomicBuiltins(M);
  // HLSL resource wrappers can keep SPIR-V handles hidden behind helper
  // `this` pointers in unoptimized IR. Inline those wrappers before lowering
  // resources so bindings are visible at the call sites.
  Changed |= markHelpersAlwaysInline(M);
  Changed |= inlineAlwaysInlineHelpers(M);
  Changed |= promoteWrapperAllocas(M);
  Changed |= lowerResourceAccess(M);
  Changed |= lowerMathBuiltins(M);
  // `markHelpersAlwaysInline` must run BEFORE `stripCallingConventions`
  // because it identifies kernels by their `SPIR_KERNEL` calling
  // convention. If we strip that first, every function looks like a
  // helper and the subsequent GlobalDCE in the target pass pipeline
  // deletes the kernels too.
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
