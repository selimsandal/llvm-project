//===-- GPUInstrInfo.cpp - GPU Instruction Info ------------===//

#include "GPUInstrInfo.h"
#include "GPUSubtarget.h"
#include "MCTargetDesc/GPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "GPUGenInstrInfo.inc"

GPUInstrInfo::GPUInstrInfo(const GPUSubtarget &STI)
    : GPUGenInstrInfo(STI, RI), RI() {}

void GPUInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MI,
                               const DebugLoc &DL, Register DestReg,
                               Register SrcReg, bool KillSrc,
                               bool RenamableDest,
                               bool RenamableSrc) const {
  BuildMI(MBB, MI, DL, get(GPU::MOV), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
}

void GPUInstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI, Register SrcReg,
    bool isKill, int FrameIndex, const TargetRegisterClass *RC,
    Register VReg, MachineInstr::MIFlag Flags) const {
  llvm_unreachable("GPU has no stack for register spilling");
}

void GPUInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        Register DestReg, int FrameIndex,
                                        const TargetRegisterClass *RC,
                                        Register VReg, unsigned SubReg,
                                        MachineInstr::MIFlag Flags) const {
  llvm_unreachable("GPU has no stack for register spilling");
}
