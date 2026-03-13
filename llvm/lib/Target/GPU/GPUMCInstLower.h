//===-- GPUMCInstLower.h - GPU MachineInstr->MCInst ---*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_GPU_GPUMCINSTLOWER_H
#define LLVM_LIB_TARGET_GPU_GPUMCINSTLOWER_H

#include "llvm/Support/Compiler.h"

namespace llvm {
class AsmPrinter;
class MachineInstr;
class MachineOperand;
class MCContext;
class MCInst;
class MCOperand;
class MCSymbol;

class GPUMCInstLower {
  MCContext &Ctx;
  const AsmPrinter &Printer;

public:
  GPUMCInstLower(MCContext &Ctx, const AsmPrinter &Printer)
      : Ctx(Ctx), Printer(Printer) {}

  void lower(const MachineInstr *MI, MCInst &OutMI) const;
  MCOperand lowerOperand(const MachineOperand &MO) const;
};

} // namespace llvm

#endif
