//===-- GPUTargetInfo.cpp - GPU Target Info ----------------===//

#include "TargetInfo/GPUTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

Target &llvm::getTheGPUTarget() {
  static Target TheGPUTarget;
  return TheGPUTarget;
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeGPUTargetInfo() {
  RegisterTarget<Triple::gpu> X(getTheGPUTarget(), "gpu",
                                "Custom GPU", "GPU");
}
