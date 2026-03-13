//===-- GPUInstrInfo.h - GPU Instruction Info -----*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_GPU_GPUINSTRINFO_H
#define LLVM_LIB_TARGET_GPU_GPUINSTRINFO_H

#include "GPURegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "GPUGenInstrInfo.inc"

namespace llvm {

class GPUSubtarget;

class GPUInstrInfo : public GPUGenInstrInfo {
  const GPURegisterInfo RI;

public:
  GPUInstrInfo(const GPUSubtarget &STI);

  const GPURegisterInfo &getRegisterInfo() const { return RI; }

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                   const DebugLoc &DL, Register DestReg, Register SrcReg,
                   bool KillSrc, bool RenamableDest = false,
                   bool RenamableSrc = false) const override;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, Register SrcReg,
                           bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC, Register VReg,
                           MachineInstr::MIFlag Flags =
                               MachineInstr::NoFlags) const override;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI, Register DestReg,
                            int FrameIndex, const TargetRegisterClass *RC,
                            Register VReg, unsigned SubReg = 0,
                            MachineInstr::MIFlag Flags =
                                MachineInstr::NoFlags) const override;
};

} // namespace llvm

#endif
