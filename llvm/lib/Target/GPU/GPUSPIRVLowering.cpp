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
// Builtin Mapping:
//   get_global_id(0)  → r0 (full thread ID = engine*8 + lane)
//   get_local_id(0)   → r0 & 7 (lane within engine)
//   get_group_id(0)   → r1 (from descriptor)
//   mad(a,b,c)        → llvm.fma.f32
//   min/max           → llvm.minnum/maxnum.f32
//
//===--------------------------------------------------------------===//

#include "GPU.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/InitializePasses.h"

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
  bool lowerMathBuiltins(Module &M);
  bool stripCallingConventions(Module &M);
  bool stripSPIRVMetadata(Module &M);
  bool removeDuplicateKernels(Module &M);
};

char GPUSPIRVLowering::ID = 0;

} // anonymous namespace

// Read a named GPU register via llvm.read_register.i32
static Value *createRegRead(IRBuilder<> &Builder, Module &M,
                            const char *RegName) {
  Function *ReadReg = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::read_register, {Builder.getInt32Ty()});
  MDNode *RegMD = MDNode::get(M.getContext(),
      {MDString::get(M.getContext(), RegName)});
  return Builder.CreateCall(ReadReg, {MetadataAsValue::get(
      M.getContext(), RegMD)});
}

// Match SPIR-V and OpenCL builtin names for thread ID functions
static int classifyBuiltin(StringRef Name) {
  // SPIR-V names (from SPIRV-LLVM-Translator metadata path)
  if (Name.contains("GlobalInvocationId")) return 1;
  if (Name.contains("WorkgroupId"))        return 2;
  if (Name.contains("LocalInvocationId"))  return 3;

  // OpenCL C mangled names (from clang -target spir round-trip)
  // _Z13get_global_idj = get_global_id(uint)
  if (Name.contains("get_global_id"))      return 1;
  // _Z12get_group_idj = get_group_id(uint)
  if (Name.contains("get_group_id"))       return 2;
  // _Z12get_local_idj = get_local_id(uint)
  if (Name.contains("get_local_id"))       return 3;

  return 0; // not a builtin
}

bool GPUSPIRVLowering::lowerBuiltinCalls(Module &M) {
  bool Changed = false;
  SmallVector<CallInst *, 16> ToReplace;
  SmallVector<Function *, 8> ToDelete;

  for (Function &F : M) {
    int Kind = classifyBuiltin(F.getName());
    if (Kind == 0)
      continue;

    for (User *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        ToReplace.push_back(CI);
    }

    for (CallInst *CI : ToReplace) {
      IRBuilder<> Builder(CI);
      Value *Result;

      if (Kind == 1) {
        // get_global_id(0) → r0 (full thread ID, not masked)
        Result = createRegRead(Builder, M, "r0");
      } else if (Kind == 2) {
        // get_group_id(0) → r1
        Result = createRegRead(Builder, M, "r1");
      } else {
        // get_local_id(0) → r0 & 7
        Value *R0 = createRegRead(Builder, M, "r0");
        Result = Builder.CreateAnd(R0, Builder.getInt32(7), "lane_id");
      }

      CI->replaceAllUsesWith(Result);
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

// Replace OpenCL math builtins with LLVM intrinsics
bool GPUSPIRVLowering::lowerMathBuiltins(Module &M) {
  bool Changed = false;
  SmallVector<CallInst *, 16> ToReplace;
  SmallVector<Function *, 8> ToDelete;

  for (Function &F : M) {
    StringRef Name = F.getName();

    // Classify math builtins
    // _Z3madfff = mad(float, float, float) → fma
    // _Z3minfff = min(float, float) → minnum
    // _Z3maxfff = max(float, float) → maxnum
    // _Z5clampffff = clamp(float, float, float) → min(max(x,lo),hi)
    enum { NONE, MAD, FMIN, FMAX } Kind = NONE;
    if (Name.starts_with("_Z3mad") || Name.starts_with("_Z4fmad"))
      Kind = MAD;
    else if (Name.starts_with("_Z3min") || Name.starts_with("_Z4fmin"))
      Kind = FMIN;
    else if (Name.starts_with("_Z3max") || Name.starts_with("_Z4fmax"))
      Kind = FMAX;

    if (Kind == NONE)
      continue;

    for (User *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        ToReplace.push_back(CI);
    }

    for (CallInst *CI : ToReplace) {
      IRBuilder<> Builder(CI);
      Value *Result = nullptr;

      if (Kind == MAD && CI->arg_size() == 3) {
        // mad(a, b, c) → fma(a, b, c)
        Function *FMA = Intrinsic::getOrInsertDeclaration(
            &M, Intrinsic::fma, {Builder.getFloatTy()});
        Result = Builder.CreateCall(FMA,
            {CI->getArgOperand(0), CI->getArgOperand(1),
             CI->getArgOperand(2)});
      } else if (Kind == FMIN && CI->arg_size() == 2) {
        Function *MinNum = Intrinsic::getOrInsertDeclaration(
            &M, Intrinsic::minnum, {Builder.getFloatTy()});
        Result = Builder.CreateCall(MinNum,
            {CI->getArgOperand(0), CI->getArgOperand(1)});
      } else if (Kind == FMAX && CI->arg_size() == 2) {
        Function *MaxNum = Intrinsic::getOrInsertDeclaration(
            &M, Intrinsic::maxnum, {Builder.getFloatTy()});
        Result = Builder.CreateCall(MaxNum,
            {CI->getArgOperand(0), CI->getArgOperand(1)});
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

  for (Function &F : M) {
    if (F.getName().starts_with("__clang_ocl_kern_imp_")) {
      ToDelete.push_back(&F);
    }
  }

  for (Function *F : ToDelete) {
    F->replaceAllUsesWith(PoisonValue::get(F->getType()));
    F->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

bool GPUSPIRVLowering::runOnModule(Module &M) {
  bool Changed = false;
  Changed |= lowerBuiltinCalls(M);
  Changed |= lowerMathBuiltins(M);
  Changed |= stripCallingConventions(M);
  Changed |= stripSPIRVMetadata(M);
  Changed |= removeDuplicateKernels(M);
  return Changed;
}

INITIALIZE_PASS(GPUSPIRVLowering, "gpu-spirv-lowering",
                "GPU SPIR-V Lowering", false, false)

namespace llvm {
ModulePass *createGPUSPIRVLoweringPass() {
  return new GPUSPIRVLowering();
}
} // namespace llvm
