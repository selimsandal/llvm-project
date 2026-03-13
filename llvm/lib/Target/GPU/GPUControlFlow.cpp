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
// Algorithm:
//   1. Un-tail-merge: Replace non-exit HALTs with branches to the exit
//      block, reconstructing the structured CFG that LLVM's optimizer
//      may have flattened via tail-merging.
//   2. Process loops (insert LOOP/ENDLOOP)
//   3. Iteratively find leaf conditionals and convert them to
//      IF/ELSE/ENDIF, then merge linear block chains so the parent
//      conditional sees a single-successor block.
//   4. Merge all MBBs into a single flat block
//   5. Compute IF/ELSE/ENDLOOP offset values from slot counts
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

struct BranchInfo {
  MachineBasicBlock *BranchTarget = nullptr;
  MachineBasicBlock *Fallthrough = nullptr;
  unsigned Invert = 0;
  unsigned FReg = 0;
};

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
  unsigned NextFlagReg = 0;

  unsigned allocFlagReg() {
    unsigned FR = NextFlagReg++;
    if (NextFlagReg > 3) NextFlagReg = 0;
    return FR;
  }

  bool getBranchInfo(MachineBasicBlock &MBB, BranchInfo &Info);
  void removeBranchPseudos(MachineBasicBlock &MBB);
  void processLoop(MachineLoop *L);
  bool processAllConditionals(MachineFunction &MF);

  void processDiamond(MachineFunction &MF, MachineBasicBlock &CondBB,
                      MachineBasicBlock *TrueBB, MachineBasicBlock *FalseBB,
                      MachineBasicBlock *MergeBB, unsigned FReg);

  void processTriangleTrue(MachineFunction &MF, MachineBasicBlock &CondBB,
                           MachineBasicBlock *ThenBB,
                           MachineBasicBlock *MergeBB, unsigned FReg);

  void processTriangleFalse(MachineFunction &MF, MachineBasicBlock &CondBB,
                            MachineBasicBlock *ThenBB,
                            MachineBasicBlock *MergeBB, unsigned FReg);

  void processSplit(MachineFunction &MF, MachineBasicBlock &CondBB,
                    MachineBasicBlock *TrueBB, MachineBasicBlock *FalseBB,
                    unsigned FReg);

  // Replace non-exit HALTs with branches to the exit block
  bool unTailMergeHALTs(MachineFunction &MF);

  // Merge a chain of linearly-connected blocks into Head
  void mergeLinearChain(MachineBasicBlock &Head);

  bool mergeAllBlocks(MachineFunction &MF);
  bool ensureHalt(MachineFunction &MF);
  void computeOffsets(MachineFunction &MF);
};

char GPUControlFlow::ID = 0;

} // anonymous namespace

INITIALIZE_PASS_BEGIN(GPUControlFlow, "gpu-control-flow",
                      "GPU Control Flow Lowering", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(GPUControlFlow, "gpu-control-flow",
                    "GPU Control Flow Lowering", false, false)

bool GPUControlFlow::getBranchInfo(MachineBasicBlock &MBB, BranchInfo &Info) {
  Info = BranchInfo();

  for (auto It = MBB.end(); It != MBB.begin();) {
    --It;
    if (It->isDebugInstr())
      continue;
    if (!It->isTerminator())
      break;
    if (It->getOpcode() == GPU::HALT)
      return false;
    if (It->getOpcode() == GPU::GPU_BR)
      Info.Fallthrough = It->getOperand(0).getMBB();
    if (It->getOpcode() == GPU::GPU_BRCOND) {
      Info.Invert = It->getOperand(0).getImm();
      Info.FReg = It->getOperand(1).getImm();
      Info.BranchTarget = It->getOperand(2).getMBB();
    }
  }

  if (!Info.BranchTarget)
    return false;
  if (!Info.Fallthrough)
    Info.Fallthrough = MBB.getNextNode();
  return Info.BranchTarget && Info.Fallthrough;
}

void GPUControlFlow::removeBranchPseudos(MachineBasicBlock &MBB) {
  auto I = MBB.end();
  while (I != MBB.begin()) {
    --I;
    if (I->isDebugInstr())
      continue;
    if (!I->isTerminator())
      break;
    if (I->getOpcode() == GPU::GPU_BRCOND ||
        I->getOpcode() == GPU::GPU_BR) {
      I->eraseFromParent();
      I = MBB.end();
    } else {
      break;
    }
  }
}

void GPUControlFlow::processLoop(MachineLoop *L) {
  for (MachineLoop *InnerLoop : *L)
    processLoop(InnerLoop);

  MachineBasicBlock *Header = L->getHeader();
  MachineBasicBlock *Latch = L->getLoopLatch();
  if (!Header || !Latch)
    return;

  unsigned FR = allocFlagReg();
  BuildMI(*Header, Header->begin(), DebugLoc(), TII->get(GPU::LOOP_INST));
  removeBranchPseudos(*Latch);
  BuildMI(*Latch, Latch->end(), DebugLoc(), TII->get(GPU::ENDLOOP_INST))
      .addImm(FR)
      .addImm(0);
}

// Reorder blocks so that A comes right after Anchor.
static void ensureAfter(MachineFunction &MF, MachineBasicBlock *Anchor,
                        MachineBasicBlock *A) {
  auto After = MachineFunction::iterator(Anchor);
  ++After;
  if (After == MF.end() || &*After != A)
    MF.splice(After, A->getIterator());
}

static void removeHALTs(MachineBasicBlock &MBB) {
  auto I = MBB.end();
  while (I != MBB.begin()) {
    --I;
    if (I->getOpcode() == GPU::HALT) {
      I->eraseFromParent();
      I = MBB.end();
    } else {
      break;
    }
  }
}

bool GPUControlFlow::unTailMergeHALTs(MachineFunction &MF) {
  // Targeted un-tail-merge: for each block ending with HALT that has
  // exactly 1 predecessor with exactly 2 successors (conditional branch),
  // replace the HALT with a branch to the other successor.
  // This reconstructs the "rejoin" edge that LLVM's tail-merge folded away.
  bool Changed = false;
  bool Progress = true;

  while (Progress) {
    Progress = false;

    for (auto &MBB : MF) {
      if (MBB.empty() || MBB.back().getOpcode() != GPU::HALT)
        continue;
      if (!MBB.succ_empty())
        continue;
      if (MBB.pred_size() != 1)
        continue;

      MachineBasicBlock *Pred = *MBB.pred_begin();
      if (Pred->succ_size() != 2)
        continue;

      // Find the other successor of the conditional predecessor
      MachineBasicBlock *OtherSucc = nullptr;
      for (auto *S : Pred->successors()) {
        if (S != &MBB) {
          OtherSucc = S;
          break;
        }
      }
      if (!OtherSucc)
        continue;

      // Don't redirect to another simple HALT-ending block with 1 predecessor
      // — that would create a cycle (both blocks redirect to each other).
      // Only redirect when the other successor is a real merge point
      // (has multiple predecessors).
      if (!OtherSucc->empty() && OtherSucc->back().getOpcode() == GPU::HALT &&
          OtherSucc->succ_empty() && OtherSucc->pred_size() <= 1)
        continue;

      removeHALTs(MBB);
      BuildMI(MBB, MBB.end(), DebugLoc(), TII->get(GPU::GPU_BR))
          .addMBB(OtherSucc);
      MBB.addSuccessor(OtherSucc);
      Changed = true;
      Progress = true;
      break; // Restart since we modified the MBB list
    }
  }

  return Changed;
}

void GPUControlFlow::mergeLinearChain(MachineBasicBlock &Head) {
  while (Head.succ_size() == 1) {
    MachineBasicBlock *Next = *Head.succ_begin();
    if (Next->pred_size() != 1)
      break;

    Head.splice(Head.end(), Next, Next->begin(), Next->end());

    Head.removeSuccessor(Next);
    SmallVector<MachineBasicBlock *, 4> Succs(Next->succ_begin(),
                                              Next->succ_end());
    for (auto *S : Succs) {
      Head.addSuccessor(S);
      Next->removeSuccessor(S);
    }

    Next->eraseFromParent();
  }
}

void GPUControlFlow::processDiamond(MachineFunction &MF,
                                    MachineBasicBlock &CondBB,
                                    MachineBasicBlock *TrueBB,
                                    MachineBasicBlock *FalseBB,
                                    MachineBasicBlock *MergeBB,
                                    unsigned FReg) {
  ensureAfter(MF, &CondBB, TrueBB);
  ensureAfter(MF, TrueBB, FalseBB);
  if (MergeBB)
    ensureAfter(MF, FalseBB, MergeBB);

  removeBranchPseudos(CondBB);
  removeBranchPseudos(*TrueBB);
  removeBranchPseudos(*FalseBB);

  // IF at end of CondBB
  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::IF_INST))
      .addImm(FReg)
      .addImm(0);

  // ELSE at end of TrueBB (after true-path code)
  BuildMI(*TrueBB, TrueBB->end(), DebugLoc(), TII->get(GPU::ELSE_INST))
      .addImm(0);

  // ENDIF at end of FalseBB (after false-path code, before merge)
  // This ensures correct nesting: inner ENDIFs come before outer ELSE/ENDIF.
  BuildMI(*FalseBB, FalseBB->end(), DebugLoc(), TII->get(GPU::ENDIF_INST));

  // Update CFG: linear chain
  while (!CondBB.succ_empty())
    CondBB.removeSuccessor(CondBB.succ_begin());
  CondBB.addSuccessor(TrueBB);

  while (!TrueBB->succ_empty())
    TrueBB->removeSuccessor(TrueBB->succ_begin());
  TrueBB->addSuccessor(FalseBB);

  while (!FalseBB->succ_empty())
    FalseBB->removeSuccessor(FalseBB->succ_begin());
  if (MergeBB)
    FalseBB->addSuccessor(MergeBB);

  // Merge the linear chain so parent conditionals see a single block
  mergeLinearChain(CondBB);
}

void GPUControlFlow::processTriangleTrue(MachineFunction &MF,
                                         MachineBasicBlock &CondBB,
                                         MachineBasicBlock *ThenBB,
                                         MachineBasicBlock *MergeBB,
                                         unsigned FReg) {
  ensureAfter(MF, &CondBB, ThenBB);
  ensureAfter(MF, ThenBB, MergeBB);

  removeBranchPseudos(CondBB);
  removeBranchPseudos(*ThenBB);

  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::IF_INST))
      .addImm(FReg)
      .addImm(0);

  // ENDIF at end of ThenBB (correct nesting: inner before outer)
  BuildMI(*ThenBB, ThenBB->end(), DebugLoc(), TII->get(GPU::ENDIF_INST));

  while (!CondBB.succ_empty())
    CondBB.removeSuccessor(CondBB.succ_begin());
  CondBB.addSuccessor(ThenBB);

  while (!ThenBB->succ_empty())
    ThenBB->removeSuccessor(ThenBB->succ_begin());
  ThenBB->addSuccessor(MergeBB);

  mergeLinearChain(CondBB);
}

void GPUControlFlow::processTriangleFalse(MachineFunction &MF,
                                          MachineBasicBlock &CondBB,
                                          MachineBasicBlock *ThenBB,
                                          MachineBasicBlock *MergeBB,
                                          unsigned FReg) {
  ensureAfter(MF, &CondBB, ThenBB);
  ensureAfter(MF, ThenBB, MergeBB);

  removeBranchPseudos(CondBB);
  removeBranchPseudos(*ThenBB);

  // IF (empty body for flag=1 lanes)
  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::IF_INST))
      .addImm(FReg)
      .addImm(0);

  // ELSE at beginning of ThenBB (flag=0 lanes execute ThenBB)
  BuildMI(*ThenBB, ThenBB->begin(), DebugLoc(), TII->get(GPU::ELSE_INST))
      .addImm(0);

  // ENDIF at end of ThenBB
  BuildMI(*ThenBB, ThenBB->end(), DebugLoc(), TII->get(GPU::ENDIF_INST));

  while (!CondBB.succ_empty())
    CondBB.removeSuccessor(CondBB.succ_begin());
  CondBB.addSuccessor(ThenBB);

  while (!ThenBB->succ_empty())
    ThenBB->removeSuccessor(ThenBB->succ_begin());
  ThenBB->addSuccessor(MergeBB);

  mergeLinearChain(CondBB);
}

void GPUControlFlow::processSplit(MachineFunction &MF,
                                  MachineBasicBlock &CondBB,
                                  MachineBasicBlock *TrueBB,
                                  MachineBasicBlock *FalseBB,
                                  unsigned FReg) {
  ensureAfter(MF, &CondBB, TrueBB);
  ensureAfter(MF, TrueBB, FalseBB);

  removeHALTs(*TrueBB);
  removeHALTs(*FalseBB);
  removeBranchPseudos(CondBB);
  removeBranchPseudos(*TrueBB);
  removeBranchPseudos(*FalseBB);

  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::IF_INST))
      .addImm(FReg)
      .addImm(0);

  BuildMI(*TrueBB, TrueBB->end(), DebugLoc(), TII->get(GPU::ELSE_INST))
      .addImm(0);

  BuildMI(*FalseBB, FalseBB->end(), DebugLoc(), TII->get(GPU::ENDIF_INST));

  while (!CondBB.succ_empty())
    CondBB.removeSuccessor(CondBB.succ_begin());
  CondBB.addSuccessor(TrueBB);

  while (!TrueBB->succ_empty())
    TrueBB->removeSuccessor(TrueBB->succ_begin());
  TrueBB->addSuccessor(FalseBB);

  while (!FalseBB->succ_empty())
    FalseBB->removeSuccessor(FalseBB->succ_begin());

  mergeLinearChain(CondBB);
}

bool GPUControlFlow::processAllConditionals(MachineFunction &MF) {
  bool Changed = false;
  bool Progress = true;

  while (Progress) {
    Progress = false;

    for (auto &MBB : MF) {
      BranchInfo Info;
      if (!getBranchInfo(MBB, Info))
        continue;

      MachineBasicBlock *TruePath =
          (Info.Invert == 0) ? Info.BranchTarget : Info.Fallthrough;
      MachineBasicBlock *FalsePath =
          (Info.Invert == 0) ? Info.Fallthrough : Info.BranchTarget;

      // Diamond: both paths converge at a common merge point
      if (TruePath->succ_size() == 1 && FalsePath->succ_size() == 1 &&
          *TruePath->succ_begin() == *FalsePath->succ_begin()) {
        MachineBasicBlock *MergeBB = *TruePath->succ_begin();
        processDiamond(MF, MBB, TruePath, FalsePath, MergeBB, Info.FReg);
        Changed = true;
        Progress = true;
        break;
      }

      // Triangle (true body): TruePath -> FalsePath
      if (TruePath->succ_size() == 1 &&
          *TruePath->succ_begin() == FalsePath) {
        processTriangleTrue(MF, MBB, TruePath, FalsePath, Info.FReg);
        Changed = true;
        Progress = true;
        break;
      }

      // Triangle (false body): FalsePath -> TruePath
      if (FalsePath->succ_size() == 1 &&
          *FalsePath->succ_begin() == TruePath) {
        processTriangleFalse(MF, MBB, FalsePath, TruePath, Info.FReg);
        Changed = true;
        Progress = true;
        break;
      }

      // Split: both paths terminate (no merge point)
      if (TruePath->succ_size() == 0 && FalsePath->succ_size() == 0) {
        processSplit(MF, MBB, TruePath, FalsePath, Info.FReg);
        Changed = true;
        Progress = true;
        break;
      }
    }
  }

  return Changed;
}

bool GPUControlFlow::mergeAllBlocks(MachineFunction &MF) {
  if (MF.size() <= 1)
    return false;

  MachineBasicBlock &EntryBB = MF.front();

  SmallVector<MachineBasicBlock *, 16> ToMerge;
  for (auto It = std::next(MF.begin()); It != MF.end(); ++It)
    ToMerge.push_back(&*It);

  for (auto *MBB : ToMerge) {
    EntryBB.splice(EntryBB.end(), MBB, MBB->begin(), MBB->end());
    while (!MBB->succ_empty())
      MBB->removeSuccessor(MBB->succ_begin());
  }

  for (auto *MBB : ToMerge)
    MBB->eraseFromParent();

  while (!EntryBB.succ_empty())
    EntryBB.removeSuccessor(EntryBB.succ_begin());

  return true;
}

bool GPUControlFlow::ensureHalt(MachineFunction &MF) {
  MachineBasicBlock &LastBB = MF.back();
  if (!LastBB.empty() && LastBB.back().getOpcode() == GPU::HALT)
    return false;
  BuildMI(LastBB, LastBB.end(), DebugLoc(), TII->get(GPU::HALT));
  return true;
}

void GPUControlFlow::computeOffsets(MachineFunction &MF) {
  DenseMap<MachineInstr *, int> SlotMap;
  int Slot = 0;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (!MI.isPseudo())
        SlotMap[&MI] = Slot++;

  struct IfEntry {
    MachineInstr *IfMI;
    MachineInstr *ElseMI = nullptr;
  };
  SmallVector<IfEntry, 8> IfStack;

  struct LoopEntry {
    MachineInstr *LoopMI;
  };
  SmallVector<LoopEntry, 4> LoopStack;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.isPseudo())
        continue;

      switch (MI.getOpcode()) {
      case GPU::IF_INST:
        IfStack.push_back({&MI});
        break;
      case GPU::ELSE_INST:
        assert(!IfStack.empty() && "ELSE without IF");
        IfStack.back().ElseMI = &MI;
        break;
      case GPU::ENDIF_INST: {
        assert(!IfStack.empty() && "ENDIF without IF");
        auto Entry = IfStack.pop_back_val();
        int EndifSlot = SlotMap[&MI];
        if (Entry.ElseMI) {
          Entry.IfMI->getOperand(1).setImm(
              SlotMap[Entry.ElseMI] - SlotMap[Entry.IfMI] - 1);
          Entry.ElseMI->getOperand(0).setImm(EndifSlot -
                                              SlotMap[Entry.ElseMI] - 1);
        } else {
          Entry.IfMI->getOperand(1).setImm(EndifSlot -
                                            SlotMap[Entry.IfMI] - 1);
        }
        break;
      }
      case GPU::LOOP_INST:
        LoopStack.push_back({&MI});
        break;
      case GPU::ENDLOOP_INST: {
        assert(!LoopStack.empty() && "ENDLOOP without LOOP");
        auto Entry = LoopStack.pop_back_val();
        MI.getOperand(1).setImm(SlotMap[Entry.LoopMI] - SlotMap[&MI]);
        break;
      }
      default:
        break;
      }
    }
  }
}

bool GPUControlFlow::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getSubtarget().getInstrInfo();
  NextFlagReg = 0;
  bool Changed = false;

  // Phase 1: Process loops
  MachineLoopInfo &MLI =
      getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  for (MachineLoop *L : MLI)
    processLoop(L);

  // Phase 2: Un-tail-merge HALTs to reconstruct structured CFG
  Changed |= unTailMergeHALTs(MF);

  // Phase 3: Convert conditionals to IF/ELSE/ENDIF
  Changed |= processAllConditionals(MF);

  // Phase 4: Remove remaining unconditional branch pseudos
  for (auto &MBB : MF) {
    auto I = MBB.end();
    while (I != MBB.begin()) {
      --I;
      if (I->isDebugInstr())
        continue;
      if (!I->isTerminator())
        break;
      if (I->getOpcode() == GPU::GPU_BR) {
        I->eraseFromParent();
        I = MBB.end();
        Changed = true;
      } else {
        break;
      }
    }
  }

  // Phase 5: Merge all MBBs into a single flat block
  Changed |= mergeAllBlocks(MF);

  // Phase 6: Ensure HALT at end
  Changed |= ensureHalt(MF);

  // Phase 7: Compute IF/ELSE/ENDLOOP offsets
  computeOffsets(MF);

  return Changed;
}

FunctionPass *llvm::createGPUControlFlowPass() {
  return new GPUControlFlow();
}
