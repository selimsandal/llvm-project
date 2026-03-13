//===-- GPUInstPrinter.h - GPU MCInst Printer ----*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_GPU_MCTARGETDESC_GPUINSTPRINTER_H
#define LLVM_LIB_TARGET_GPU_MCTARGETDESC_GPUINSTPRINTER_H

#include "llvm/MC/MCInstPrinter.h"

namespace llvm {

class GPUInstPrinter : public MCInstPrinter {
public:
  GPUInstPrinter(const MCAsmInfo &MAI, const MCInstrInfo &MII,
                 const MCRegisterInfo &MRI)
      : MCInstPrinter(MAI, MII, MRI) {}

  void printInst(const MCInst *MI, uint64_t Address, StringRef Annot,
                 const MCSubtargetInfo &STI, raw_ostream &O) override;
  void printRegName(raw_ostream &O, MCRegister Reg) override;
  void printOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O);
  void printImm32Operand(const MCInst *MI, unsigned OpNo, raw_ostream &O);
  void printBrTargetOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O);
  void printCondCodeOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O);

  // Auto-generated
  std::pair<const char *, uint64_t> getMnemonic(const MCInst &MI) const override;
  void printInstruction(const MCInst *MI, uint64_t Address, raw_ostream &O);
  static const char *getRegisterName(MCRegister Reg);
};

} // namespace llvm

#endif
