//===-- GPUTargetMachine.cpp - GPU Target Machine ---------===//

#include "GPUTargetMachine.h"
#include "GPU.h"
#include "GPUSubtarget.h"
#include "TargetInfo/GPUTargetInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"

using namespace llvm;

static const char *GPUDataLayout = "e-p:32:32-i32:32-f32:32-n32";

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::Static);
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeGPUTarget() {
  RegisterTargetMachine<GPUTargetMachine> X(getTheGPUTarget());
  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeGPUDAGToDAGISelLegacyPass(PR);
  initializeGPUControlFlowPass(PR);
  initializeGPUPeepholePass(PR);
}

GPUTargetMachine::GPUTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   std::optional<Reloc::Model> RM,
                                   std::optional<CodeModel::Model> CM,
                                   CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, GPUDataLayout, TT,
                               CPU.empty() ? "generic" : CPU, FS, Options,
                               getEffectiveRelocModel(RM),
                               CM.value_or(CodeModel::Small), OL),
      Subtarget(TT, CPU.empty() ? "generic" : std::string(CPU),
                std::string(FS), *this),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {
  initAsmInfo();
}

namespace {
class GPUPassConfig : public TargetPassConfig {
public:
  GPUPassConfig(GPUTargetMachine &TM, PassManagerBase *PM)
      : TargetPassConfig(TM, *PM) {}

  GPUTargetMachine &getGPUTargetMachine() const {
    return getTM<GPUTargetMachine>();
  }

  bool addInstSelector() override {
    addPass(createGPUISelDag(getGPUTargetMachine(), getOptLevel()));
    return false;
  }

  void addPreEmitPass() override {
    addPass(createGPUControlFlowPass());
    addPass(createGPUPeepholePass());
  }
};
} // namespace

TargetPassConfig *GPUTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new GPUPassConfig(*this, &PM);
}
