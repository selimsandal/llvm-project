//===-- GPUTargetMachine.h - GPU Target Machine ---*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_GPU_GPUTARGETMACHINE_H
#define LLVM_LIB_TARGET_GPU_GPUTARGETMACHINE_H

#include "GPUSubtarget.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"

namespace llvm {

class GPUTargetMachine : public CodeGenTargetMachineImpl {
  GPUSubtarget Subtarget;
  std::unique_ptr<TargetLoweringObjectFile> TLOF;

public:
  GPUTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                   StringRef FS, const TargetOptions &Options,
                   std::optional<Reloc::Model> RM,
                   std::optional<CodeModel::Model> CM,
                   CodeGenOptLevel OL, bool JIT);

  const GPUSubtarget *getSubtargetImpl(const Function &F) const override {
    return &Subtarget;
  }

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
};

} // namespace llvm

#endif
