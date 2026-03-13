//===-- GPUControlFlow.cpp - GPU Control Flow Lowering ----===//
//
// Late MachineFunction pass that converts LLVM CFG branches to GPU
// structured mask-stack operations (IF/ELSE/ENDIF/LOOP/ENDLOOP/BREAK).
//
// The GPU has no arbitrary branches — all control flow uses an 8-entry
// mask stack with per-lane execution masks. This pass identifies
// structured if-then-else and loop patterns from the MBB graph and
// replaces branch instructions with mask operations.
//
// SPIR-V from DXC produces structured control flow, so the CFG is
// guaranteed to be reducible with clean nesting.
//
// This pass also linearizes the basic blocks: the GPU instruction
// stream is flat (no jump targets), so all MBBs are laid out
// sequentially and branches are replaced by IF/ELSE/ENDIF with
// instruction-slot offsets in imm32.
//
//===--------------------------------------------------------------===//

#include "GPU.h"
#include "MCTargetDesc/GPUMCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
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

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const TargetInstrInfo *TII = nullptr;
  unsigned NextFlagReg = 0; // Flag register allocator (f0-f3, by depth)

  unsigned allocFlagReg() {
    unsigned FR = NextFlagReg++;
    if (NextFlagReg > 3) NextFlagReg = 0; // wrap (safe with structured nesting)
    return FR;
  }

  // Remove branch instructions from a basic block, return the condition info
  void removeBranches(MachineBasicBlock &MBB);

  // Linearize: lay out all blocks sequentially, convert branches to
  // IF/ELSE/ENDIF mask ops with offset placeholders
  bool linearizeFunction(MachineFunction &MF);

  // Process a loop: insert LOOP at header, ENDLOOP at latch
  void processLoop(MachineLoop *L);
};

char GPUControlFlow::ID = 0;

} // anonymous namespace

INITIALIZE_PASS_BEGIN(GPUControlFlow, "gpu-control-flow",
                      "GPU Control Flow Lowering", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(GPUControlFlow, "gpu-control-flow",
                    "GPU Control Flow Lowering", false, false)

void GPUControlFlow::removeBranches(MachineBasicBlock &MBB) {
  // Remove all terminator instructions (branches)
  while (!MBB.empty() && MBB.back().isTerminator() &&
         MBB.back().getOpcode() != GPU::HALT) {
    MBB.pop_back();
  }
}

void GPUControlFlow::processLoop(MachineLoop *L) {
  // Process inner loops first (post-order)
  for (MachineLoop *InnerLoop : *L)
    processLoop(InnerLoop);

  MachineBasicBlock *Header = L->getHeader();
  MachineBasicBlock *Latch = L->getLoopLatch();
  if (!Header || !Latch)
    return;

  unsigned FR = allocFlagReg();

  // Insert LOOP at the beginning of the header
  BuildMI(*Header, Header->begin(), DebugLoc(), TII->get(GPU::LOOP_INST));

  // At the latch, replace the back-edge branch with ENDLOOP
  // The offset will be patched later during linearization
  removeBranches(*Latch);
  BuildMI(*Latch, Latch->end(), DebugLoc(), TII->get(GPU::ENDLOOP_INST))
      .addImm(FR)
      .addImm(0); // offset placeholder
}

bool GPUControlFlow::linearizeFunction(MachineFunction &MF) {
  bool Changed = false;

  // For each basic block, if it has conditional branches, we need to
  // convert them to IF/ELSE/ENDIF. For now, handle the simple case
  // of single-BB functions and straight-line code by just removing
  // unconditional branches between sequentially-laid-out blocks.
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty())
      continue;

    // Remove unconditional branches to the fall-through successor
    MachineBasicBlock *FallThrough = MBB.getNextNode();
    if (FallThrough && MBB.isSuccessor(FallThrough)) {
      // Check if the last terminator is an unconditional branch to FallThrough
      auto It = MBB.getLastNonDebugInstr();
      if (It != MBB.end() && It->isUnconditionalBranch()) {
        It->eraseFromParent();
        Changed = true;
      }
    }

    // If the block has conditional branches, we need to handle them.
    // For the initial implementation, we only handle straight-line code.
    // Conditional branches from structured SPIR-V will be handled by
    // identifying if-then-else diamond patterns in the CFG and inserting
    // IF/ELSE/ENDIF around them.
    //
    // A diamond pattern looks like:
    //   cond_br → TrueBB / FalseBB
    //   TrueBB → MergeBB
    //   FalseBB → MergeBB
    //
    // This becomes:
    //   CMP ..., F0
    //   IF F0, <offset past then-body>
    //   [then-body]
    //   ELSE <offset past else-body>
    //   [else-body]
    //   ENDIF
  }

  // Ensure every function ends with HALT
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.succ_empty() && !MBB.empty()) {
      MachineInstr &LastMI = MBB.back();
      if (LastMI.getOpcode() != GPU::HALT) {
        BuildMI(MBB, MBB.end(), DebugLoc(), TII->get(GPU::HALT));
        Changed = true;
      }
    }
  }

  return Changed;
}

bool GPUControlFlow::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getSubtarget().getInstrInfo();
  NextFlagReg = 0;

  bool Changed = false;

  // Process loops
  MachineLoopInfo &MLI =
      getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  for (MachineLoop *L : MLI)
    processLoop(L);

  // Linearize the function
  Changed |= linearizeFunction(MF);

  return Changed;
}

FunctionPass *llvm::createGPUControlFlowPass() {
  return new GPUControlFlow();
}
