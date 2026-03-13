//===-- GPUMCTargetDesc.cpp - GPU MC Target Desc ----------===//

#include "GPUMCTargetDesc.h"
#include "GPUMCAsmInfo.h"
#include "GPUInstPrinter.h"
#include "TargetInfo/GPUTargetInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "GPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "GPUGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "GPUGenRegisterInfo.inc"

static MCInstrInfo *createGPUMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitGPUMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createGPUMCRegisterInfo(const Triple & /*TT*/) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitGPUMCRegisterInfo(X, GPU::R5); // RA = R5 (arbitrary, no calls)
  return X;
}

static MCSubtargetInfo *
createGPUMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  std::string CPUName = std::string(CPU);
  if (CPUName.empty())
    CPUName = "generic";
  return createGPUMCSubtargetInfoImpl(TT, CPUName, CPUName, FS);
}

static MCInstPrinter *createGPUMCInstPrinter(const Triple & /*T*/,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new GPUInstPrinter(MAI, MII, MRI);
  return nullptr;
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeGPUTargetMC() {
  Target &T = getTheGPUTarget();

  RegisterMCAsmInfo<GPUMCAsmInfo> X(T);
  TargetRegistry::RegisterMCInstrInfo(T, createGPUMCInstrInfo);
  TargetRegistry::RegisterMCRegInfo(T, createGPUMCRegisterInfo);
  TargetRegistry::RegisterMCSubtargetInfo(T, createGPUMCSubtargetInfo);
  TargetRegistry::RegisterMCCodeEmitter(T, createGPUMCCodeEmitter);
  TargetRegistry::RegisterMCAsmBackend(T, createGPUAsmBackend);
  TargetRegistry::RegisterMCInstPrinter(T, createGPUMCInstPrinter);
}
