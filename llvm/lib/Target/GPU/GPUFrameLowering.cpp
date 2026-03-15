//===-- GPUFrameLowering.cpp - GPU Frame Lowering ---------===//

#include "GPUFrameLowering.h"
#include "GPUSubtarget.h"
#include "MCTargetDesc/GPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

using namespace llvm;

GPUFrameLowering::GPUFrameLowering(const GPUSubtarget &STI)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown,
                          Align(4), 0) {}

void GPUFrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  if (!MFI.getNumObjects())
    return;  // No spills — zero overhead

  // Compute frame size: number of spill slots * 4 bytes per slot
  uint64_t FrameSize = MFI.estimateStackSize(MF);
  if (FrameSize == 0)
    return;

  // Round up to multiple of 4
  FrameSize = (FrameSize + 3) & ~3ULL;
  MFI.setStackSize(FrameSize);

  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL;

  // R30 = R0 * frame_size  (per-lane stack offset)
  BuildMI(MBB, MBBI, DL, TII.get(GPU::MULi), GPU::R30)
      .addReg(GPU::R0)
      .addImm(FrameSize);

  // R30 = R30 + spill_base  (0x00380000 = DDR spill area)
  BuildMI(MBB, MBBI, DL, TII.get(GPU::ADDi), GPU::R30)
      .addReg(GPU::R30)
      .addImm(0x00380000);
}

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
