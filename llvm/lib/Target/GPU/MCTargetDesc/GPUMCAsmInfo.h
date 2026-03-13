//===-- GPUMCAsmInfo.h - GPU Asm Info --------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_GPU_MCTARGETDESC_GPUMCASMINFO_H
#define LLVM_LIB_TARGET_GPU_MCTARGETDESC_GPUMCASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
class Triple;

class GPUMCAsmInfo : public MCAsmInfo {
public:
  explicit GPUMCAsmInfo(const Triple &TT, const MCTargetOptions &Options);
};

} // namespace llvm

#endif
