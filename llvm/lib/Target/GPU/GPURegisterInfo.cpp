//===-- GPURegisterInfo.cpp - GPU Register Info ------------===//

#include "GPURegisterInfo.h"
#include "GPUFrameLowering.h"
#include "GPUSubtarget.h"
#include "MCTargetDesc/GPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "GPUGenRegisterInfo.inc"

GPURegisterInfo::GPURegisterInfo()
    : GPUGenRegisterInfo(GPU::R5) {} // Return address reg (arbitrary)

const MCPhysReg *
GPURegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  // No function calls, no callee-saved regs needed
  static const MCPhysReg CalleeSavedRegs[] = {0};
  return CalleeSavedRegs;
}

BitVector GPURegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(GPU::R0);  // Thread ID (read-only, auto-initialized)
  Reserved.set(GPU::R30); // Stack pointer
  return Reserved;
}

bool GPURegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                          int SPAdj, unsigned FIOperandNum,
                                          RegScavenger *RS) const {
  MachineInstr &Inst = *MI;
  MachineBasicBlock &MBB = *Inst.getParent();
  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  DebugLoc DL = Inst.getDebugLoc();

  int FrameIndex = Inst.getOperand(FIOperandNum).getIndex();
  int Offset = MF.getFrameInfo().getObjectOffset(FrameIndex) +
               MF.getFrameInfo().getStackSize();

  unsigned Opc = Inst.getOpcode();
  if (Opc == GPU::SPILL_GPR) {
    Register SrcReg = Inst.getOperand(0).getReg();
    BuildMI(MBB, MI, DL, TII.get(GPU::ST_SCATTER))
        .addReg(GPU::R30)
        .addReg(SrcReg)
        .addImm(Offset);
    Inst.eraseFromParent();
  } else if (Opc == GPU::RELOAD_GPR) {
    Register DstReg = Inst.getOperand(0).getReg();
    BuildMI(MBB, MI, DL, TII.get(GPU::LD_SCATTER), DstReg)
        .addReg(GPU::R30)
        .addImm(Offset);
    Inst.eraseFromParent();
  } else {
    llvm_unreachable("Unexpected instruction with frame index");
  }
  return false;
}

Register GPURegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  // Stack pointer register
  return GPU::R30;
}
