//===-- GPUSubtarget.cpp - GPU Subtarget Info --------------===//

#include "GPUSubtarget.h"
#include "MCTargetDesc/GPUMCTargetDesc.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "GPUGenSubtargetInfo.inc"

void GPUSubtarget::initSubtargetFeatures(StringRef CPU, StringRef FS) {
  std::string CPUName = std::string(CPU);
  if (CPUName.empty())
    CPUName = "generic";
  ParseSubtargetFeatures(CPUName, CPUName, FS);
}

GPUSubtarget &GPUSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                             StringRef FS) {
  initSubtargetFeatures(CPU, FS);
  return *this;
}

GPUSubtarget::GPUSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                           const TargetMachine &TM)
    : GPUGenSubtargetInfo(TT, CPU, CPU, FS),
      FrameLowering(initializeSubtargetDependencies(CPU, FS)),
      InstrInfo(*this),
      TLInfo(TM, *this) {}
