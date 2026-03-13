//===-- GPUMCAsmInfo.cpp - GPU Asm Info --------------------===//

#include "GPUMCAsmInfo.h"

using namespace llvm;

GPUMCAsmInfo::GPUMCAsmInfo(const Triple &TT, const MCTargetOptions &Options) {
  IsLittleEndian = true;
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;
  SupportsDebugInformation = false;
  CommentString = "//";
  // GPU outputs flat binary, no ELF
  HasFunctionAlignment = false;
}
