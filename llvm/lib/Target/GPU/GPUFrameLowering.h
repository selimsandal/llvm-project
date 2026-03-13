//===-- GPUFrameLowering.h - GPU Frame Lowering ---*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_GPU_GPUFRAMELOWERING_H
#define LLVM_LIB_TARGET_GPU_GPUFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

class GPUSubtarget;

class GPUFrameLowering : public TargetFrameLowering {
public:
  GPUFrameLowering(const GPUSubtarget &STI);

  void emitPrologue(MachineFunction &MF,
                    MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF,
                    MachineBasicBlock &MBB) const override;
  bool hasFPImpl(const MachineFunction &MF) const override;
  MachineBasicBlock::iterator
  eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const override;
};

} // namespace llvm

#endif
