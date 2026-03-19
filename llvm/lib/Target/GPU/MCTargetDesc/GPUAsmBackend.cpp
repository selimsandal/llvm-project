//===-- GPUAsmBackend.cpp - GPU Assembler Backend --------===//
//
// Minimal AsmBackend for ELF container output.
// .text contains raw 128-bit GPU instructions.
// Optional sidecar sections such as .gpu.meta carry launch/reflection data.
//
//===--------------------------------------------------------------===//

#include "MCTargetDesc/GPUMCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

class GPUELFObjectWriter : public MCELFObjectTargetWriter {
public:
  GPUELFObjectWriter()
      : MCELFObjectTargetWriter(/*Is64Bit=*/false, /*OSABI=*/0,
                                /*EMachine=*/ELF::EM_NONE,
                                /*HasRelocationAddend=*/false) {}

  unsigned getRelocType(const MCFixup &Fixup, const MCValue &Target,
                        bool IsPCRel) const override {
    return 0;
  }
};

class GPUAsmBackend : public MCAsmBackend {
public:
  GPUAsmBackend(const MCSubtargetInfo &STI)
      : MCAsmBackend(llvm::endianness::little) {}

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    return std::make_unique<GPUELFObjectWriter>();
  }

  void applyFixup(const MCFragment &, const MCFixup &, const MCValue &Target,
                  uint8_t *Data, uint64_t Value, bool IsResolved) override {
    // No relocations — all addresses are absolute
  }

  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override {
    // GPU NOP is 16 bytes (128-bit instruction, all zeros)
    for (uint64_t i = 0; i < Count; ++i)
      OS.write('\0');
    return true;
  }
};

} // anonymous namespace

MCAsmBackend *llvm::createGPUAsmBackend(const Target &T,
                                        const MCSubtargetInfo &STI,
                                        const MCRegisterInfo &MRI,
                                        const MCTargetOptions &Options) {
  return new GPUAsmBackend(STI);
}
