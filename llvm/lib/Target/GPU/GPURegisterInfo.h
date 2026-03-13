//===-- GPURegisterInfo.h - GPU Register Info -----*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_GPU_GPUREGISTERINFO_H
#define LLVM_LIB_TARGET_GPU_GPUREGISTERINFO_H

#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "GPUGenRegisterInfo.inc"

namespace llvm {

class GPURegisterInfo : public GPUGenRegisterInfo {
public:
  GPURegisterInfo();

  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;
  BitVector getReservedRegs(const MachineFunction &MF) const override;
  bool eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;
  Register getFrameRegister(const MachineFunction &MF) const override;
};

} // namespace llvm

#endif
