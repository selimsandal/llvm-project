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
  bool foldImmediates(MachineBasicBlock &MBB);
  bool foldSourceModifiers(MachineBasicBlock &MBB);
  void computeOffsets(MachineFunction &MF);

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

      // Check that SrcReg isn't used AFTER the FADD either — deleting
      // the FMUL would leave those later uses reading a stale value.
      {
        bool UsedAfter = false;
        auto It = std::next(MachineBasicBlock::iterator(&MI));
        for (; It != MBB.end(); ++It) {
          // Check uses before defs: an instruction like "r13 = op r14, r13"
          // both uses and defines r13 — the use reads the OLD value.
          bool HasUse = false, HasDef = false;
          for (const MachineOperand &MO : It->operands()) {
            if (MO.isReg() && MO.getReg() == SrcReg) {
              if (MO.isUse()) HasUse = true;
              if (MO.isDef()) HasDef = true;
            }
          }
          if (HasUse) { UsedAfter = true; break; }
          if (HasDef) break; // Redefined without reading old value
        }
        if (UsedAfter)
          continue;
      }

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

// Map register-form opcode to its immediate-form counterpart.
// Returns 0 if no immediate form exists.
static unsigned getImmediateOpcode(unsigned Opc) {
  switch (Opc) {
  case GPU::ADD:  return GPU::ADDi;
  case GPU::SUB:  return GPU::SUBi;
  case GPU::MUL:  return GPU::MULi;
  case GPU::AND:  return GPU::ANDi;
  case GPU::OR:   return GPU::ORi;
  case GPU::XOR:  return GPU::XORi;
  case GPU::SHL:  return GPU::SHLi;
  case GPU::SHR:  return GPU::SHRi;
  case GPU::SHRA: return GPU::SHRAi;
  case GPU::FADD: return GPU::FADDi;
  case GPU::FMUL: return GPU::FMULi;
  case GPU::FSUB: return GPU::FSUBi;
  default:        return 0;
  }
}

// MOVI(t, imm) + ALU(d, x, t) -> ALUi(d, x, imm)
// when t has no other uses between the MOVI and the ALU op.
bool GPUPeephole::foldImmediates(MachineBasicBlock &MBB) {
  bool Changed = false;
  SmallVector<std::tuple<MachineInstr *, MachineInstr *, bool>, 4> ToFold;

  for (MachineInstr &MI : MBB) {
    unsigned ImmOpc = getImmediateOpcode(MI.getOpcode());
    if (!ImmOpc)
      continue;

    // Check if src1 (operand 2) was defined by a nearby MOVI
    if (!MI.getOperand(2).isReg())
      continue;
    Register Src1Reg = MI.getOperand(2).getReg();

    MachineInstr *MoviMI = findSingleDef(MBB, MI.getIterator(), Src1Reg);
    if (!MoviMI || MoviMI->getOpcode() != GPU::MOVI)
      continue;

    if (MoviMI->getOperand(0).getReg() != Src1Reg)
      continue;

    if (hasInterveningUse(MBB, MoviMI, &MI, Src1Reg))
      continue;

    // Check if Src1Reg is used AFTER the ALU op — if so, keep the MOVI
    bool UsedAfter = false;
    {
      auto It = std::next(MachineBasicBlock::iterator(&MI));
      for (; It != MBB.end(); ++It) {
        bool HasUse = false, HasDef = false;
        for (const MachineOperand &MO : It->operands()) {
          if (MO.isReg() && MO.getReg() == Src1Reg) {
            if (MO.isUse()) HasUse = true;
            if (MO.isDef()) HasDef = true;
          }
        }
        if (HasUse) { UsedAfter = true; break; }
        if (HasDef) break;
      }
    }

    // Found: MOVI(t, imm) + ALU(d, x, t) -> ALUi(d, x, imm)
    uint32_t ImmVal = MoviMI->getOperand(1).getImm();
    Register DstReg = MI.getOperand(0).getReg();
    Register Src0Reg = MI.getOperand(1).getReg();

    const TargetInstrInfo &TII =
        *MBB.getParent()->getSubtarget().getInstrInfo();

    BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(ImmOpc), DstReg)
        .addReg(Src0Reg)
        .addImm(ImmVal);

    ToFold.push_back({MoviMI, &MI, UsedAfter});
    Changed = true;
  }

  for (auto &[MoviMI, AluMI, KeepMovi] : ToFold) {
    AluMI->eraseFromParent();
    if (!KeepMovi)
      MoviMI->eraseFromParent();
  }

  return Changed;
}

// Source modifier folding: FSUB(R31, x) -> FNEG pattern, AND(x, 0x7FFFFFFF) -> FABS pattern.
// When the result feeds a float ALU op as src1, we can fold by
// changing the ALU src1 to x and emitting with src1_mod bits.
// Since we can't add mod operands to existing instructions without
// new TableGen defs, we annotate the MI's operand TargetFlags.
// The MCCodeEmitter reads these flags and sets src_mod bits.
//
// TargetFlags on a register operand:
//   0 = no modifier
//   1 = NEG (flip sign bit)
//   2 = ABS (clear sign bit)
//   3 = NEGABS (set sign bit)
//
// This folds: FSUB(zero, x) + FADD(d, y, t) -> FADD(d, y, x) with src1_mod=1
static bool isFloatALU(unsigned Opc) {
  switch (Opc) {
  case GPU::FADD: case GPU::FMUL: case GPU::FSUB:
  case GPU::FDIV: case GPU::FMIN: case GPU::FMAX:
    return true;
  default:
    return false;
  }
}

bool GPUPeephole::foldSourceModifiers(MachineBasicBlock &MBB) {
  bool Changed = false;
  SmallVector<MachineInstr *, 4> ToErase;

  for (MachineInstr &MI : MBB) {
    if (!isFloatALU(MI.getOpcode()))
      continue;

    // Check src0 (operand 1) and src1 (operand 2) for modifier patterns
    for (unsigned SrcIdx = 1; SrcIdx <= 2; ++SrcIdx) {
      if (!MI.getOperand(SrcIdx).isReg())
        continue;
      Register SrcReg = MI.getOperand(SrcIdx).getReg();

      MachineInstr *DefMI = findSingleDef(MBB, MI.getIterator(), SrcReg);
      if (!DefMI)
        continue;

      unsigned Mod = 0;
      Register RealSrc;

      // Pattern: FSUB(MOVI(0), x) -> NEG(x)
      if (DefMI->getOpcode() == GPU::FSUB &&
          DefMI->getOperand(1).isReg()) {
        Register FSUBSrc0 = DefMI->getOperand(1).getReg();
        MachineInstr *Src0Def = findSingleDef(MBB, DefMI->getIterator(), FSUBSrc0);
        if (Src0Def && Src0Def->getOpcode() == GPU::MOVI &&
            Src0Def->getOperand(1).getImm() == 0 &&
            !hasInterveningUse(MBB, Src0Def, DefMI, FSUBSrc0)) {
          // Only erase MOVI if the zero reg isn't used after the FSUB
          bool ZeroUsedAfter = false;
          auto ZIt = std::next(MachineBasicBlock::iterator(DefMI));
          for (; ZIt != MBB.end(); ++ZIt) {
            for (const MachineOperand &MO : ZIt->operands()) {
              if (MO.isReg() && MO.isUse() && MO.getReg() == FSUBSrc0)
                ZeroUsedAfter = true;
              if (MO.isReg() && MO.isDef() && MO.getReg() == FSUBSrc0)
                goto zero_scan_done;
            }
          }
          zero_scan_done:
          Mod = 1; // NEG
          RealSrc = DefMI->getOperand(2).getReg();
          if (!ZeroUsedAfter)
            ToErase.push_back(Src0Def);
        }
      }
      // Pattern: ANDi(x, 0x7FFFFFFF) -> ABS(x)
      else if (DefMI->getOpcode() == GPU::ANDi &&
               DefMI->getOperand(2).getImm() == 0x7FFFFFFF) {
        Mod = 2; // ABS
        RealSrc = DefMI->getOperand(1).getReg();
      }

      if (Mod == 0)
        continue;

      // Check the def result is only used by this instruction
      if (DefMI->getOperand(0).getReg() != SrcReg)
        continue;
      if (hasInterveningUse(MBB, DefMI, &MI, SrcReg))
        continue;

      // Fold: replace the source register and set target flags
      MI.getOperand(SrcIdx).setReg(RealSrc);
      MI.getOperand(SrcIdx).setTargetFlags(Mod);
      ToErase.push_back(DefMI);
      Changed = true;
    }
  }

  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();

  return Changed;
}

// Compute IF/ELSE/ENDLOOP/BREAK offsets from final instruction slots.
// Must run after all instruction-count-changing optimizations (FMA, imm fold).
void GPUPeephole::computeOffsets(MachineFunction &MF) {
  DenseMap<MachineInstr *, int> SlotMap;
  int Slot = 0;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (!MI.isPseudo())
        SlotMap[&MI] = Slot++;

  struct IfEntry {
    MachineInstr *IfMI;
    MachineInstr *ElseMI = nullptr;
  };
  SmallVector<IfEntry, 8> IfStack;

  struct LoopEntry {
    MachineInstr *LoopMI;
    SmallVector<MachineInstr *, 2> Breaks;
  };
  SmallVector<LoopEntry, 4> LoopStack;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.isPseudo())
        continue;

      switch (MI.getOpcode()) {
      case GPU::IF_INST:
        IfStack.push_back({&MI});
        break;
      case GPU::ELSE_INST:
        assert(!IfStack.empty() && "ELSE without IF");
        IfStack.back().ElseMI = &MI;
        break;
      case GPU::ENDIF_INST: {
        assert(!IfStack.empty() && "ENDIF without IF");
        auto Entry = IfStack.pop_back_val();
        int EndifSlot = SlotMap[&MI];
        if (Entry.ElseMI) {
          // Hardware IF/ELSE skip: pc = pc_plus1 + imm32
          Entry.IfMI->getOperand(1).setImm(
              SlotMap[Entry.ElseMI] - SlotMap[Entry.IfMI] - 1);
          Entry.ElseMI->getOperand(0).setImm(EndifSlot -
                                              SlotMap[Entry.ElseMI] - 1);
        } else {
          Entry.IfMI->getOperand(1).setImm(EndifSlot -
                                            SlotMap[Entry.IfMI] - 1);
        }
        break;
      }
      case GPU::LOOP_INST:
        LoopStack.push_back({&MI, {}});
        break;
      case GPU::BREAK_INST:
        assert(!LoopStack.empty() && "BREAK without LOOP");
        LoopStack.back().Breaks.push_back(&MI);
        break;
      case GPU::ENDLOOP_INST: {
        assert(!LoopStack.empty() && "ENDLOOP without LOOP");
        auto Entry = LoopStack.pop_back_val();
        // Hardware ENDLOOP: pc = pc_plus1 + imm32. Target = LOOP + 1 (after LOOP push).
        MI.getOperand(1).setImm(SlotMap[Entry.LoopMI] - SlotMap[&MI]);
        // BREAK: pc = pc_plus1 + imm32. Target = ENDLOOP slot.
        int EndloopSlot = SlotMap[&MI];
        for (auto *BreakMI : Entry.Breaks)
          BreakMI->getOperand(1).setImm(EndloopSlot - SlotMap[BreakMI] - 1);
        break;
      }
      default:
        break;
      }
    }
  }
}

bool GPUPeephole::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    Changed |= formFMA(MBB);
    Changed |= foldImmediates(MBB);
    Changed |= foldSourceModifiers(MBB);
    Changed |= eliminateRedundantMov(MBB);
  }

  // Compute control flow offsets after all instruction-count-changing
  // optimizations, so offsets reflect final instruction positions.
  computeOffsets(MF);

  return Changed;
}

FunctionPass *llvm::createGPUPeepholePass() {
  return new GPUPeephole();
}
