//===-- GPUControlFlow.cpp - GPU Control Flow Lowering ----===//
//
// Late MachineFunction pass that converts LLVM branches to GPU
// structured mask-stack operations (IF/ELSE/ENDIF/LOOP/ENDLOOP/BREAK).
//
// Since SPIR-V from DXC produces structured control flow, we can
// identify if-then-else and loop patterns directly from the CFG.
//
// This pass runs after register allocation and before final emission.
//
//===--------------------------------------------------------------===//

#include "GPU.h"
#include "GPUSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-control-flow"

namespace {

class GPUControlFlow : public MachineFunctionPass {
public:
  static char ID;
  GPUControlFlow() : MachineFunctionPass(ID) {
    initializeGPUControlFlowPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "GPU Control Flow Lowering";
  }
};

char GPUControlFlow::ID = 0;

} // anonymous namespace

INITIALIZE_PASS(GPUControlFlow, "gpu-control-flow",
                "GPU Control Flow Lowering", false, false)

bool GPUControlFlow::runOnMachineFunction(MachineFunction &MF) {
  // Phase 2 implementation stub.
  // For MVP, all code is straight-line (no branches).
  // SPIR-V structured control flow will be lowered here in Phase 2.
  //
  // The pass will:
  // 1. Identify structured if-then-else patterns and emit IF/ELSE/ENDIF
  // 2. Identify loop patterns and emit LOOP/ENDLOOP
  // 3. Handle break statements with BREAK
  // 4. Compute branch offsets (two-pass: emit then patch imm32)
  // 5. Allocate flag registers by nesting depth

  bool Changed = false;

  // For now, ensure every function ends with HALT
  for (MachineBasicBlock &MBB : MF) {
    if (!MBB.empty()) {
      MachineInstr &LastMI = MBB.back();
      // Check if last instruction is already HALT
      if (LastMI.getOpcode() == GPU::HALT)
        continue;
      // Check if this is the last basic block with a terminator
      if (MBB.succ_empty() && !LastMI.isTerminator()) {
        BuildMI(MBB, MBB.end(), DebugLoc(), MF.getSubtarget().getInstrInfo()->get(GPU::HALT));
        Changed = true;
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createGPUControlFlowPass() {
  return new GPUControlFlow();
}
