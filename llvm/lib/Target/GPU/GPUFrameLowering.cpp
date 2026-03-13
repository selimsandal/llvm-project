//===-- GPUFrameLowering.cpp - GPU Frame Lowering ---------===//

#include "GPUFrameLowering.h"
#include "GPUSubtarget.h"
#include "llvm/CodeGen/MachineFunction.h"

using namespace llvm;

GPUFrameLowering::GPUFrameLowering(const GPUSubtarget &STI)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown,
                          Align(4), 0) {}

void GPUFrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {}

void GPUFrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {}

bool GPUFrameLowering::hasFPImpl(const MachineFunction &MF) const {
  return false;
}

MachineBasicBlock::iterator GPUFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MI) const {
  return MBB.erase(MI);
}
