//===-- GPUPeephole.cpp - GPU Peephole Optimizations -----===//
//
// Post-RA peephole pass for:
// - FMA formation (FMUL + FADD → FMA when register is single-def-use)
// - Redundant MOV elimination
//
// These run after register allocation, so we work with physical
// registers and scan locally within basic blocks.
//
//===--------------------------------------------------------------===//

#include "GPU.h"
#include "MCTargetDesc/GPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-peephole"

namespace {

class GPUPeephole : public MachineFunctionPass {
public:
  static char ID;
  GPUPeephole() : MachineFunctionPass(ID) {
    initializeGPUPeepholePass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "GPU Peephole Optimizations";
  }

private:
  bool formFMA(MachineBasicBlock &MBB);
  bool eliminateRedundantMov(MachineBasicBlock &MBB);

  // Find the unique instruction in MBB that defines PhysReg,
  // between Start (exclusive, searching backward) and the beginning.
  // Returns nullptr if not found or multiple defs.
  MachineInstr *findSingleDef(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator Before,
                              Register PhysReg);

  // Check if PhysReg is used between DefMI (exclusive) and UseMI (exclusive)
  bool hasInterveningUse(MachineBasicBlock &MBB,
                         MachineInstr *DefMI, MachineInstr *UseMI,
                         Register PhysReg);
};

char GPUPeephole::ID = 0;

} // anonymous namespace

INITIALIZE_PASS(GPUPeephole, "gpu-peephole",
                "GPU Peephole Optimizations", false, false)

MachineInstr *GPUPeephole::findSingleDef(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator Before,
                                          Register PhysReg) {
  // Scan backward from Before to find the def of PhysReg
  auto It = Before;
  while (It != MBB.begin()) {
    --It;
    MachineInstr &MI = *It;

    // Check if this instruction defines PhysReg
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg() == PhysReg)
        return &MI;
    }

    // If something clobbers PhysReg implicitly, stop
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isUse() && MO.getReg() == PhysReg)
        return nullptr; // Used before def found — can't fold
    }
  }
  return nullptr;
}

bool GPUPeephole::hasInterveningUse(MachineBasicBlock &MBB,
                                    MachineInstr *DefMI, MachineInstr *UseMI,
                                    Register PhysReg) {
  // Check if PhysReg is used between DefMI and UseMI (exclusive of both)
  auto It = MachineBasicBlock::iterator(DefMI);
  ++It;
  auto End = MachineBasicBlock::iterator(UseMI);

  for (; It != End; ++It) {
    for (const MachineOperand &MO : It->operands()) {
      if (MO.isReg() && MO.getReg() == PhysReg &&
          (MO.isUse() || MO.isDef()))
        return true;
    }
  }
  return false;
}

// FMUL(t, a, b) + FADD(d, t, c) → FMA(d, a, b, c)
// when t is not used elsewhere between the two instructions
bool GPUPeephole::formFMA(MachineBasicBlock &MBB) {
  bool Changed = false;
  SmallVector<std::pair<MachineInstr *, MachineInstr *>, 4> ToReplace;

  for (MachineInstr &MI : MBB) {
    if (MI.getOpcode() != GPU::FADD)
      continue;

    // Check each source operand to see if it comes from FMUL
    for (unsigned SrcIdx = 1; SrcIdx <= 2; ++SrcIdx) {
      if (!MI.getOperand(SrcIdx).isReg())
        continue;
      Register SrcReg = MI.getOperand(SrcIdx).getReg();

      MachineInstr *MulMI = findSingleDef(MBB, MI.getIterator(), SrcReg);
      if (!MulMI || MulMI->getOpcode() != GPU::FMUL)
        continue;

      // Check that SrcReg isn't used between FMUL and FADD
      if (hasInterveningUse(MBB, MulMI, &MI, SrcReg))
        continue;

      // Check that FMUL's dst == SrcReg
      if (MulMI->getOperand(0).getReg() != SrcReg)
        continue;

      // Found the pattern: FMUL(t, a, b) + FADD(d, t, c)
      unsigned OtherIdx = (SrcIdx == 1) ? 2 : 1;
      Register AddendReg = MI.getOperand(OtherIdx).getReg();
      Register DstReg = MI.getOperand(0).getReg();
      Register MulSrc0 = MulMI->getOperand(1).getReg();
      Register MulSrc1 = MulMI->getOperand(2).getReg();

      const TargetInstrInfo &TII =
          *MBB.getParent()->getSubtarget().getInstrInfo();

      BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(GPU::FMA), DstReg)
          .addReg(MulSrc0)
          .addReg(MulSrc1)
          .addReg(AddendReg);

      ToReplace.push_back({MulMI, &MI});
      Changed = true;
      break;
    }
  }

  for (auto &[MulMI, AddMI] : ToReplace) {
    AddMI->eraseFromParent();
    MulMI->eraseFromParent();
  }

  return Changed;
}

bool GPUPeephole::eliminateRedundantMov(MachineBasicBlock &MBB) {
  bool Changed = false;
  SmallVector<MachineInstr *, 4> ToErase;

  for (MachineInstr &MI : MBB) {
    if (MI.getOpcode() != GPU::MOV)
      continue;
    if (MI.getOperand(0).getReg() == MI.getOperand(1).getReg()) {
      ToErase.push_back(&MI);
      Changed = true;
    }
  }

  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();

  return Changed;
}

bool GPUPeephole::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    Changed |= formFMA(MBB);
    Changed |= eliminateRedundantMov(MBB);
  }

  return Changed;
}

FunctionPass *llvm::createGPUPeepholePass() {
  return new GPUPeephole();
}
