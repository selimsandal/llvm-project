//===-- GPUTargetInfo.h - GPU Target Info ----------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_GPU_TARGETINFO_GPUTARGETINFO_H
#define LLVM_LIB_TARGET_GPU_TARGETINFO_GPUTARGETINFO_H

namespace llvm {
class Target;
Target &getTheGPUTarget();
} // namespace llvm

#endif
