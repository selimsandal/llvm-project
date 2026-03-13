//===-- GPUMCInstLower.cpp - GPU MachineInstr->MCInst -----===//

#include "GPUMCInstLower.h"
#include "MCTargetDesc/GPUMCTargetDesc.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"

using namespace llvm;

MCOperand GPUMCInstLower::lowerOperand(const MachineOperand &MO) const {
  switch (MO.getType()) {
  default:
    llvm_unreachable("Unknown operand type");
  case MachineOperand::MO_Register:
    if (MO.isImplicit())
      return MCOperand();
    return MCOperand::createReg(MO.getReg());
  case MachineOperand::MO_Immediate:
    return MCOperand::createImm(MO.getImm());
  case MachineOperand::MO_GlobalAddress: {
    const MCSymbol *Sym = Printer.getSymbol(MO.getGlobal());
    const MCExpr *Expr = MCSymbolRefExpr::create(Sym, Ctx);
    if (MO.getOffset())
      Expr = MCBinaryExpr::createAdd(
          Expr, MCConstantExpr::create(MO.getOffset(), Ctx), Ctx);
    return MCOperand::createExpr(Expr);
  }
  case MachineOperand::MO_MachineBasicBlock:
    return MCOperand::createExpr(
        MCSymbolRefExpr::create(MO.getMBB()->getSymbol(), Ctx));
  }
}

static bool isFloatALUOpcode(unsigned Opc) {
  switch (Opc) {
  case GPU::FADD: case GPU::FMUL: case GPU::FSUB:
  case GPU::FDIV: case GPU::FMIN: case GPU::FMAX:
  case GPU::FMA:
    return true;
  default:
    return false;
  }
}

void GPUMCInstLower::lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  // For float ALU ops, extract source modifier TargetFlags from register
  // operands and pack into MCInst flags for the encoder.
  // Encoding: flags[1:0] = src0_mod, flags[3:2] = src1_mod
  if (isFloatALUOpcode(MI->getOpcode())) {
    unsigned Src0Mod = 0, Src1Mod = 0;
    // src0 is operand index 1, src1 is operand index 2
    if (MI->getNumOperands() > 1 && MI->getOperand(1).isReg())
      Src0Mod = MI->getOperand(1).getTargetFlags() & 0x3;
    if (MI->getNumOperands() > 2 && MI->getOperand(2).isReg())
      Src1Mod = MI->getOperand(2).getTargetFlags() & 0x3;
    OutMI.setFlags(Src0Mod | (Src1Mod << 2));
  }

  for (const MachineOperand &MO : MI->operands()) {
    MCOperand MCOp = lowerOperand(MO);
    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}
