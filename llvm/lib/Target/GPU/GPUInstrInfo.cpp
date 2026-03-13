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

static bool isGPUBranch(unsigned Opc) {
  return Opc == GPU::GPU_BRCOND || Opc == GPU::GPU_BR;
}

bool GPUInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                 MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  TBB = nullptr;
  FBB = nullptr;
  Cond.clear();

  // Walk backward from the end of the block looking for terminators.
  auto I = MBB.end();
  while (I != MBB.begin()) {
    --I;
    if (I->isDebugInstr())
      continue;
    if (!I->isTerminator())
      break;

    if (I->getOpcode() == GPU::HALT)
      return true; // Can't analyze block ending with HALT

    if (I->getOpcode() == GPU::GPU_BR) {
      if (!TBB) {
        TBB = I->getOperand(0).getMBB();
        continue;
      }
      return true; // Multiple unconditional branches
    }

    if (I->getOpcode() == GPU::GPU_BRCOND) {
      MachineBasicBlock *Target = I->getOperand(2).getMBB();
      if (TBB) {
        // GPU_BRCOND followed by GPU_BR: conditional + fallback
        FBB = TBB;
        TBB = Target;
        Cond.push_back(I->getOperand(0)); // invert
        Cond.push_back(I->getOperand(1)); // freg
        return false;
      }
      // Just a conditional branch (fallthrough to next MBB)
      TBB = Target;
      Cond.push_back(I->getOperand(0)); // invert
      Cond.push_back(I->getOperand(1)); // freg
      return false;
    }

    return true; // Unknown terminator
  }
  return false;
}

unsigned GPUInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                    int *BytesRemoved) const {
  unsigned Count = 0;
  auto I = MBB.end();
  while (I != MBB.begin()) {
    --I;
    if (I->isDebugInstr())
      continue;
    if (!I->isTerminator())
      break;
    if (!isGPUBranch(I->getOpcode()))
      break;
    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }
  if (BytesRemoved)
    *BytesRemoved = Count * 16; // 128-bit instructions
  return Count;
}

unsigned GPUInstrInfo::insertBranch(MachineBasicBlock &MBB,
                                    MachineBasicBlock *TBB,
                                    MachineBasicBlock *FBB,
                                    ArrayRef<MachineOperand> Cond,
                                    const DebugLoc &DL,
                                    int *BytesAdded) const {
  unsigned Count = 0;

  if (Cond.empty()) {
    // Unconditional branch
    BuildMI(&MBB, DL, get(GPU::GPU_BR)).addMBB(TBB);
    ++Count;
  } else {
    // Conditional branch: Cond[0]=invert, Cond[1]=freg
    BuildMI(&MBB, DL, get(GPU::GPU_BRCOND))
        .addImm(Cond[0].getImm())
        .addImm(Cond[1].getImm())
        .addMBB(TBB);
    ++Count;
  }

  if (FBB) {
    BuildMI(&MBB, DL, get(GPU::GPU_BR)).addMBB(FBB);
    ++Count;
  }

  if (BytesAdded)
    *BytesAdded = Count * 16;
  return Count;
}

bool GPUInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 2);
  Cond[0].setImm(!Cond[0].getImm()); // flip invert flag
  return false;
}
