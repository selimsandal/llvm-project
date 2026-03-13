//===-- GPURegisterInfo.cpp - GPU Register Info ------------===//

#include "GPURegisterInfo.h"
#include "GPUFrameLowering.h"
#include "GPUSubtarget.h"
#include "MCTargetDesc/GPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunction.h"
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
  Reserved.set(GPU::R31); // Hardwired zero
  return Reserved;
}

bool GPURegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                          int SPAdj, unsigned FIOperandNum,
                                          RegScavenger *RS) const {
  // GPU has no stack - this should never be called
  llvm_unreachable("GPU has no stack frame");
  return false;
}

Register GPURegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  // No frame pointer
  return GPU::R31;
}
