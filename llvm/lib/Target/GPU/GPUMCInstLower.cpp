//===-- GPUMCInstLower.cpp - GPU MachineInstr->MCInst -----===//

#include "GPUMCInstLower.h"
#include "GPU.h"
#include "MCTargetDesc/GPUMCTargetDesc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"

using namespace llvm;

namespace {

DenseMap<const MachineInstr *, unsigned> GPUSourceModifierMap;

} // namespace

void llvm::setGPUSourceModifier(const MachineInstr *MI, unsigned SrcIdx,
                                unsigned Mod) {
  assert(SrcIdx < 3 && "unexpected GPU source-modifier operand index");
  assert((Mod & ~0x3u) == 0 && "GPU source modifier must fit in 2 bits");
  unsigned Shift = SrcIdx * 2;
  unsigned &Packed = GPUSourceModifierMap[MI];
  Packed = (Packed & ~(0x3u << Shift)) | ((Mod & 0x3u) << Shift);
}

unsigned llvm::takeGPUSourceModifiers(const MachineInstr *MI) {
  auto It = GPUSourceModifierMap.find(MI);
  if (It == GPUSourceModifierMap.end())
    return 0;
  unsigned Packed = It->second;
  GPUSourceModifierMap.erase(It);
  return Packed;
}

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

  // For float ALU ops, extract the packed source modifiers recorded by the
  // peephole pass and hand them to the encoder in MCInst flags.
  // Encoding: flags[1:0] = src0_mod, flags[3:2] = src1_mod
  if (isFloatALUOpcode(MI->getOpcode())) {
    OutMI.setFlags(takeGPUSourceModifiers(MI));
  }

  for (const MachineOperand &MO : MI->operands()) {
    MCOperand MCOp = lowerOperand(MO);
    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}
