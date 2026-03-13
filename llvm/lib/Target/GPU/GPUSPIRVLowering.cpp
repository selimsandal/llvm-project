//===-- GPUSPIRVLowering.cpp - SPIR-V → GPU Lowering -----===//
//
// IR-level pass that lowers SPIR-V builtins to GPU register reads
// and maps SPIR-V buffer accesses to scatter/gather addresses.
//
// This runs on LLVM IR after SPIRV-LLVM-Translator and before ISel.
//
// SPIR-V Builtin Mapping:
//   SV_DispatchThreadID.x  → r0 & 7 (lane ID within SIMD-8)
//   SV_GroupID.x           → r1     (from descriptor init_r1)
//   SV_GroupThreadID.x     → r0 & 7 (same as lane ID)
//
// SPIR-V Storage Class Mapping:
//   StorageBuffer/UAV      → LD_SCATTER/ST_SCATTER (base via r1-r4)
//   Uniform/CBV            → init_r1-r4 (small) or LDV (large)
//   Workgroup (shared)     → Not supported (no LDS)
//
//===--------------------------------------------------------------===//

#include "GPU.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
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
  bool lowerBufferAccesses(Module &M);
};

char GPUSPIRVLowering::ID = 0;

} // anonymous namespace

// Look for calls to SPIR-V builtin functions and replace them
// with GPU register reads.
//
// SPIRV-LLVM-Translator produces calls like:
//   @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0)
//   @_Z28__spirv_BuiltInWorkgroupIdi(i32 0)
//   @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0)
//
// We replace these with inline asm reads of r0/r1.
bool GPUSPIRVLowering::lowerBuiltinCalls(Module &M) {
  bool Changed = false;
  SmallVector<CallInst *, 16> ToReplace;

  for (Function &F : M) {
    StringRef Name = F.getName();

    // Match SPIR-V builtin function names
    bool IsGlobalInvocationId = Name.contains("GlobalInvocationId");
    bool IsWorkgroupId = Name.contains("WorkgroupId");
    bool IsLocalInvocationId = Name.contains("LocalInvocationId");

    if (!IsGlobalInvocationId && !IsWorkgroupId && !IsLocalInvocationId)
      continue;

    for (User *U : F.users()) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI) continue;
      ToReplace.push_back(CI);
    }

    for (CallInst *CI : ToReplace) {
      IRBuilder<> Builder(CI);

      Value *Result;
      if (IsGlobalInvocationId || IsLocalInvocationId) {
        // SV_DispatchThreadID.x = r0 & 7 (lane index)
        // Read r0 via inline asm
        FunctionType *AsmTy = FunctionType::get(
            Builder.getInt32Ty(), false);
        InlineAsm *ReadR0 = InlineAsm::get(
            AsmTy, "mov $0, r0", "=r", /*hasSideEffects=*/true);
        Value *R0 = Builder.CreateCall(AsmTy, ReadR0);
        Result = Builder.CreateAnd(R0, Builder.getInt32(7), "lane_id");
      } else {
        // SV_GroupID.x = r1 (from descriptor init)
        FunctionType *AsmTy = FunctionType::get(
            Builder.getInt32Ty(), false);
        InlineAsm *ReadR1 = InlineAsm::get(
            AsmTy, "mov $0, r1", "=r", /*hasSideEffects=*/true);
        Value *R1 = Builder.CreateCall(AsmTy, ReadR1);
        Result = R1;
      }

      CI->replaceAllUsesWith(Result);
      CI->eraseFromParent();
      Changed = true;
    }
    ToReplace.clear();
  }

  return Changed;
}

// For now, buffer accesses are handled by the standard ISel patterns.
// SPIR-V buffer loads/stores become GEP + load/store in LLVM IR,
// which the ISel lowers to LD_SCATTER/ST_SCATTER.
bool GPUSPIRVLowering::lowerBufferAccesses(Module &M) {
  // Future: strip SPIR-V metadata, handle buffer binding descriptors
  return false;
}

bool GPUSPIRVLowering::runOnModule(Module &M) {
  bool Changed = false;
  Changed |= lowerBuiltinCalls(M);
  Changed |= lowerBufferAccesses(M);
  return Changed;
}

INITIALIZE_PASS(GPUSPIRVLowering, "gpu-spirv-lowering",
                "GPU SPIR-V Lowering", false, false)

namespace llvm {
ModulePass *createGPUSPIRVLoweringPass() {
  return new GPUSPIRVLowering();
}
} // namespace llvm
