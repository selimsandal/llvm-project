//===-- GPU.h - Top-level interface for GPU target ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the
// GPU target library, as used by the LLVM JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_GPU_GPU_H
#define LLVM_LIB_TARGET_GPU_GPU_H

#include "MCTargetDesc/GPUMCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class GPUTargetMachine;
class FunctionPass;
class ModulePass;
class PassRegistry;

FunctionPass *createGPUISelDag(GPUTargetMachine &TM,
                               CodeGenOptLevel OptLevel);

void initializeGPUDAGToDAGISelLegacyPass(PassRegistry &);

FunctionPass *createGPUControlFlowPass();
void initializeGPUControlFlowPass(PassRegistry &);

FunctionPass *createGPUPeepholePass();
void initializeGPUPeepholePass(PassRegistry &);

ModulePass *createGPUSPIRVLoweringPass();
void initializeGPUSPIRVLoweringPass(PassRegistry &);

} // namespace llvm

#endif
