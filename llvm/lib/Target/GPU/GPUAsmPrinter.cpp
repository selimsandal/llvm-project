//===-- GPUAsmPrinter.cpp - GPU Assembly Printer ----------===//
//
// Emits GPU machine code into the ELF object streamer.
// .text contains raw 128-bit GPU instructions packed two per 256-bit beat.
// The object may also carry auxiliary sections such as .gpu.meta emitted by
// earlier IR passes.
//
//===--------------------------------------------------------------===//

#include "GPU.h"
#include "GPUMCInstLower.h"
#include "GPUSubtarget.h"
#include "GPUTargetMachine.h"
#include "TargetInfo/GPUTargetInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-asm-printer"

namespace {

class GPUAsmPrinter : public AsmPrinter {
public:
  explicit GPUAsmPrinter(TargetMachine &TM,
                         std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override { return "GPU Assembly Printer"; }

  void emitInstruction(const MachineInstr *MI) override;
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // anonymous namespace

bool GPUAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  SetupMachineFunction(MF);
  emitFunctionBody();
  return false;
}

void GPUAsmPrinter::emitInstruction(const MachineInstr *MI) {
  // Skip pseudo instructions that shouldn't be emitted
  if (MI->isPseudo())
    return;

  GPUMCInstLower Lower(OutContext, *this);
  MCInst TmpInst;
  Lower.lower(MI, TmpInst);
  EmitToStreamer(*OutStreamer, TmpInst);
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeGPUAsmPrinter() {
  RegisterAsmPrinter<GPUAsmPrinter> X(getTheGPUTarget());
}
