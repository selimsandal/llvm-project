//===-- GPUMCTargetDesc.h - GPU MC Target Desc ---*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_GPU_MCTARGETDESC_GPUMCTARGETDESC_H
#define LLVM_LIB_TARGET_GPU_MCTARGETDESC_GPUMCTARGETDESC_H

#include <memory>

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCTargetOptions;
class Target;

MCCodeEmitter *createGPUMCCodeEmitter(const MCInstrInfo &MCII,
                                      MCContext &Ctx);

MCAsmBackend *createGPUAsmBackend(const Target &T,
                                  const MCSubtargetInfo &STI,
                                  const MCRegisterInfo &MRI,
                                  const MCTargetOptions &Options);
} // namespace llvm

// Defines symbolic names for GPU registers.
#define GET_REGINFO_ENUM
#include "GPUGenRegisterInfo.inc"

// Defines symbolic names for GPU instructions.
#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "GPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "GPUGenSubtargetInfo.inc"

#endif
