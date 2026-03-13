//===-- GPUSubtarget.h - GPU Subtarget Def -------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_GPU_GPUSUBTARGET_H
#define LLVM_LIB_TARGET_GPU_GPUSUBTARGET_H

#include "GPUFrameLowering.h"
#include "GPUISelLowering.h"
#include "GPUInstrInfo.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

#define GET_SUBTARGETINFO_HEADER
#include "GPUGenSubtargetInfo.inc"

namespace llvm {

class GPUSubtarget : public GPUGenSubtargetInfo {
  GPUFrameLowering FrameLowering;
  GPUInstrInfo InstrInfo;
  GPUTargetLowering TLInfo;
  SelectionDAGTargetInfo TSInfo;

  // Subtarget features
  bool HasScoreboard = false;
  bool HasBarrelThreading = false;

  void initSubtargetFeatures(StringRef CPU, StringRef FS);
  GPUSubtarget &initializeSubtargetDependencies(StringRef CPU, StringRef FS);

public:
  GPUSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
               const TargetMachine &TM);

  void ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS);

  const GPUInstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const GPUFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  const GPURegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }
  const GPUTargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }
  const SelectionDAGTargetInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }

  bool hasScoreboard() const { return HasScoreboard; }
  bool hasBarrelThreading() const { return HasBarrelThreading; }
};

} // namespace llvm

#endif
