//===-- GPUMCInstLower.cpp - GPU MachineInstr->MCInst -----===//

#include "GPUMCInstLower.h"
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

void GPUMCInstLower::lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  for (const MachineOperand &MO : MI->operands()) {
    MCOperand MCOp = lowerOperand(MO);
    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}
