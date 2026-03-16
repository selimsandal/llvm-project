//===-- GPUControlFlow.cpp - GPU Control Flow Lowering ----===//
//
// Late MachineFunction pass that converts LLVM CFG branches to GPU
// GOTO/JOIN/WHILE/BREAK/JUMP mask-stack operations.
//
// The GPU has no arbitrary branches — all control flow uses an 8-entry
// mask stack with per-lane execution masks. This pass identifies
// structured if-then-else and loop patterns from the MBB graph and
// replaces branch instructions with mask operations.
//
// Control flow model:
//   GOTO(flag, JIP):  push disabled lanes, remove from EM, jump if EM=0
//   JOIN:             pop stack, reactivate parked lanes
//   WHILE:            push empty entry with is_loop=1 (loop init)
//   BREAK(flag, tgt): accumulate into nearest loop entry, jump if EM=0
//   JUMP(tgt):        unconditional branch (loop back-edge)
//
// Algorithm:
//   1. Process loops (insert WHILE/BREAK/JUMP/JOIN)
//   2. Un-tail-merge HALTs
//   3. Convert conditionals to GOTO/JOIN
//   4. Merge all MBBs into a single flat block
//   5. Ensure trailing HALT
//   (Offset computation deferred to GPUPeephole)
//
//===--------------------------------------------------------------===//

#include "GPU.h"
#include "GPURegisterInfo.h"
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

  bool unTailMergeHALTs(MachineFunction &MF);
  void mergeLinearChain(MachineBasicBlock &Head);
  bool mergeAllBlocks(MachineFunction &MF);
  bool ensureHalt(MachineFunction &MF);

  // Map flag register number (0-3) to physical register (F0-F3)
  unsigned flagReg(unsigned N) const {
    static const unsigned FRegs[] = {GPU::F0, GPU::F1, GPU::F2, GPU::F3};
    return FRegs[N & 3];
  }

  // Compute pred mode for BREAK/GOTO given branch semantics.
  // WantExitFlag: true if flag=1 should mean "take the branch"
  // ExitIsTarget: true if the exit goes to BranchTarget (not Fallthrough)
  // Invert: the BRCOND invert field
  // Returns 1 (PRED_IF) or 2 (PRED_IF_NOT)
  unsigned computePredMode(bool ExitIsTarget, unsigned Invert) const {
    // BRCOND: "if flag[FReg] != Invert, goto BranchTarget"
    //   Invert=0: flag=1 → branch, flag=0 → fallthrough
    //   Invert=1: flag=0 → branch, flag=1 → fallthrough
    //
    // For BREAK/GOTO, pred_mask determines which lanes are disabled:
    //   PRED_IF (1): flag value used as-is → flag=1 lanes removed
    //   PRED_IF_NOT (2): ~flag used → flag=0 lanes removed
    //
    // We want "exit lanes" to be removed. Exit is via:
    //   ExitIsTarget=true: lanes that would branch → flag=1 (Inv=0), flag=0 (Inv=1)
    //   ExitIsTarget=false: lanes that would fall through → flag=0 (Inv=0), flag=1 (Inv=1)
    if (ExitIsTarget)
      return (Invert == 0) ? 1 : 2;  // PRED_IF : PRED_IF_NOT
    else
      return (Invert == 0) ? 2 : 1;  // PRED_IF_NOT : PRED_IF
  }
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

  // Insert WHILE at header start (push empty loop entry)
  BuildMI(*Header, Header->begin(), DebugLoc(), TII->get(GPU::WHILE_INST));

  // Handle early loop exits (BREAK) in non-latch blocks
  for (MachineBasicBlock *BB : L->getBlocks()) {
    if (BB == Latch)
      continue;

    BranchInfo Info;
    if (!getBranchInfo(*BB, Info))
      continue;

    bool TargetInLoop = L->contains(Info.BranchTarget);
    bool FallthroughInLoop = L->contains(Info.Fallthrough);

    if (TargetInLoop == FallthroughInLoop)
      continue;

    MachineBasicBlock *ExitBB =
        TargetInLoop ? Info.Fallthrough : Info.BranchTarget;
    MachineBasicBlock *InLoopBB =
        TargetInLoop ? Info.BranchTarget : Info.Fallthrough;
    bool ExitIsTarget = !TargetInLoop;

    unsigned BreakPred = computePredMode(ExitIsTarget, Info.Invert);

    removeBranchPseudos(*BB);

    // BREAK: accumulate exiting lanes into loop's stack entry.
    // No exit-block cloning needed — BREAK removes lanes from EM, freezing
    // their register values. JOIN reactivates them with correct per-lane state.
    BuildMI(*BB, BB->end(), DebugLoc(), TII->get(GPU::BREAK_INST))
        .addReg(flagReg(Info.FReg))
        .addImm(BreakPred)
        .addImm(0);  // offset placeholder

    // Update CFG
    BB->removeSuccessor(ExitBB);
    if (!BB->isSuccessor(InLoopBB))
      BB->addSuccessor(InLoopBB);

    // Clear exit block if unreachable
    if (ExitBB->pred_empty()) {
      SmallVector<MachineInstr *, 8> ToDel;
      for (auto &MI : *ExitBB)
        ToDel.push_back(&MI);
      for (auto *MI : ToDel)
        MI->eraseFromParent();
    }
  }

  // Handle latch: BREAK (exiting lanes) + JUMP (continuing lanes back to top)
  {
    BranchInfo LatchBI;
    if (getBranchInfo(*Latch, LatchBI)) {
      bool TargetInLoop = L->contains(LatchBI.BranchTarget);
      unsigned BreakPred = computePredMode(!TargetInLoop, LatchBI.Invert);

      removeBranchPseudos(*Latch);

      // BREAK for exiting lanes
      BuildMI(*Latch, Latch->end(), DebugLoc(), TII->get(GPU::BREAK_INST))
          .addReg(flagReg(LatchBI.FReg))
          .addImm(BreakPred)
          .addImm(0);  // offset placeholder

      // JUMP back to loop body (instruction after WHILE)
      BuildMI(*Latch, Latch->end(), DebugLoc(), TII->get(GPU::JUMP_INST))
          .addImm(0);  // offset placeholder
    } else {
      // Unconditional latch (infinite loop or single-exit via break)
      removeBranchPseudos(*Latch);
      BuildMI(*Latch, Latch->end(), DebugLoc(), TII->get(GPU::JUMP_INST))
          .addImm(0);
    }
  }

  // Insert JOIN at the first exit block (reactivates accumulated lanes)
  SmallVector<MachineBasicBlock *, 2> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  if (!ExitBlocks.empty()) {
    MachineBasicBlock *ExitBB = ExitBlocks[0];
    BuildMI(*ExitBB, ExitBB->begin(), DebugLoc(), TII->get(GPU::JOIN_INST));
  }

  // Update latch CFG: remove back-edge, keep exit edge
  SmallVector<MachineBasicBlock *, 4> LatchSuccsCopy(Latch->succ_begin(),
                                                      Latch->succ_end());
  for (auto *S : LatchSuccsCopy) {
    if (L->contains(S))
      Latch->removeSuccessor(S);
  }

  // Merge loop-internal blocks (not exit block)
  while (Header->succ_size() == 1) {
    MachineBasicBlock *Next = *Header->succ_begin();
    if (Next->pred_size() != 1)
      break;
    if (!L->contains(Next))
      break;
    Header->splice(Header->end(), Next, Next->begin(), Next->end());
    Header->removeSuccessor(Next);
    SmallVector<MachineBasicBlock *, 4> Succs(Next->succ_begin(),
                                              Next->succ_end());
    for (auto *S : Succs) {
      Header->addSuccessor(S);
      Next->removeSuccessor(S);
    }
    Next->eraseFromParent();
  }
}

// Reorder blocks so A comes right after Anchor.
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

      MachineBasicBlock *OtherSucc = nullptr;
      for (auto *S : Pred->successors()) {
        if (S != &MBB) {
          OtherSucc = S;
          break;
        }
      }
      if (!OtherSucc)
        continue;

      if (!OtherSucc->empty() && OtherSucc->back().getOpcode() == GPU::HALT &&
          OtherSucc->succ_empty() && OtherSucc->pred_size() <= 1)
        continue;

      removeHALTs(MBB);
      BuildMI(MBB, MBB.end(), DebugLoc(), TII->get(GPU::GPU_BR))
          .addMBB(OtherSucc);
      MBB.addSuccessor(OtherSucc);
      Changed = true;
      Progress = true;
      break;
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

// Diamond: if/else/endif → GOTO/JOIN/GOTO/JOIN
//   GOTO(flag, else) → push true-lanes, skip if all false
//   ... true body ...
//   JOIN             → reactivate true-lanes
//   GOTO(~flag, end) → push false-lanes, skip if all true
//   ... false body ...
//   JOIN             → reactivate false-lanes
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

  // GOTO: flag=1 lanes (false-path) get pushed, true-path lanes stay active
  // BRCOND flag=1 branches to TrueBB. For GOTO, we push the "other" lanes.
  // Since TrueBB is the flag=1 path, we want flag=0 lanes pushed → PRED_IF_NOT
  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(FReg))
      .addImm(2)   // PRED_IF_NOT: push flag=0 (false) lanes, true lanes active
      .addImm(0);  // JIP offset placeholder

  // JOIN at end of TrueBB (reactivate false-path lanes)
  BuildMI(*TrueBB, TrueBB->end(), DebugLoc(), TII->get(GPU::JOIN_INST));

  // GOTO: now push true-path lanes (flag=1), false-path lanes active
  BuildMI(*TrueBB, TrueBB->end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(FReg))
      .addImm(1)   // PRED_IF: push flag=1 (true) lanes, false lanes active
      .addImm(0);

  // JOIN at end of FalseBB (reactivate true-path lanes)
  BuildMI(*FalseBB, FalseBB->end(), DebugLoc(), TII->get(GPU::JOIN_INST));

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

  mergeLinearChain(CondBB);
}

// Triangle (true body): flag=1 → execute ThenBB, flag=0 → skip
//   GOTO(~flag, merge) → push flag=0 lanes, true lanes active
//   ... then body ...
//   JOIN               → reactivate flag=0 lanes
void GPUControlFlow::processTriangleTrue(MachineFunction &MF,
                                         MachineBasicBlock &CondBB,
                                         MachineBasicBlock *ThenBB,
                                         MachineBasicBlock *MergeBB,
                                         unsigned FReg) {
  ensureAfter(MF, &CondBB, ThenBB);
  ensureAfter(MF, ThenBB, MergeBB);

  removeBranchPseudos(CondBB);
  removeBranchPseudos(*ThenBB);

  // GOTO: push flag=0 (false/skip) lanes, flag=1 (then) lanes stay active
  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(FReg))
      .addImm(2)   // PRED_IF_NOT: flag=0 lanes pushed
      .addImm(0);

  // JOIN at end of ThenBB
  BuildMI(*ThenBB, ThenBB->end(), DebugLoc(), TII->get(GPU::JOIN_INST));

  while (!CondBB.succ_empty())
    CondBB.removeSuccessor(CondBB.succ_begin());
  CondBB.addSuccessor(ThenBB);

  while (!ThenBB->succ_empty())
    ThenBB->removeSuccessor(ThenBB->succ_begin());
  ThenBB->addSuccessor(MergeBB);

  mergeLinearChain(CondBB);
}

// Triangle (false body): flag=0 → execute ThenBB, flag=1 → skip
//   GOTO(flag, merge) → push flag=1 lanes, flag=0 lanes active
//   ... body ...
//   JOIN              → reactivate flag=1 lanes
void GPUControlFlow::processTriangleFalse(MachineFunction &MF,
                                          MachineBasicBlock &CondBB,
                                          MachineBasicBlock *ThenBB,
                                          MachineBasicBlock *MergeBB,
                                          unsigned FReg) {
  ensureAfter(MF, &CondBB, ThenBB);
  ensureAfter(MF, ThenBB, MergeBB);

  removeBranchPseudos(CondBB);
  removeBranchPseudos(*ThenBB);

  // GOTO: push flag=1 lanes, flag=0 lanes stay active for the body
  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(FReg))
      .addImm(1)   // PRED_IF: flag=1 lanes pushed
      .addImm(0);

  // JOIN at end of body
  BuildMI(*ThenBB, ThenBB->end(), DebugLoc(), TII->get(GPU::JOIN_INST));

  while (!CondBB.succ_empty())
    CondBB.removeSuccessor(CondBB.succ_begin());
  CondBB.addSuccessor(ThenBB);

  while (!ThenBB->succ_empty())
    ThenBB->removeSuccessor(ThenBB->succ_begin());
  ThenBB->addSuccessor(MergeBB);

  mergeLinearChain(CondBB);
}

// Split: both paths terminate
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

  // Same as diamond but no merge point
  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(FReg))
      .addImm(2)   // PRED_IF_NOT
      .addImm(0);

  BuildMI(*TrueBB, TrueBB->end(), DebugLoc(), TII->get(GPU::JOIN_INST));

  BuildMI(*TrueBB, TrueBB->end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(FReg))
      .addImm(1)   // PRED_IF
      .addImm(0);

  BuildMI(*FalseBB, FalseBB->end(), DebugLoc(), TII->get(GPU::JOIN_INST));

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

      // Diamond: both paths converge
      if (TruePath->succ_size() == 1 && FalsePath->succ_size() == 1 &&
          *TruePath->succ_begin() == *FalsePath->succ_begin()) {
        MachineBasicBlock *MergeBB = *TruePath->succ_begin();

        // Multi-predecessor FalseBB: redirect to triangle to fix nesting
        if (FalsePath->pred_size() > 1) {
          MBB.removeSuccessor(FalsePath);
          if (!MBB.isSuccessor(MergeBB))
            MBB.addSuccessor(MergeBB);

          if (FalsePath->pred_size() == 1) {
            MachineBasicBlock *Pred = *FalsePath->pred_begin();
            if (Pred->succ_size() == 1)
              mergeLinearChain(*Pred);
          }

          processTriangleTrue(MF, MBB, TruePath, MergeBB, Info.FReg);
          Changed = true;
          Progress = true;
          break;
        }

        processDiamond(MF, MBB, TruePath, FalsePath, MergeBB, Info.FReg);
        Changed = true;
        Progress = true;
        break;
      }

      // Triangle (true body)
      if (TruePath->succ_size() == 1 &&
          *TruePath->succ_begin() == FalsePath) {
        processTriangleTrue(MF, MBB, TruePath, FalsePath, Info.FReg);
        Changed = true;
        Progress = true;
        break;
      }

      // Triangle (false body)
      if (FalsePath->succ_size() == 1 &&
          *FalsePath->succ_begin() == TruePath) {
        processTriangleFalse(MF, MBB, FalsePath, TruePath, Info.FReg);
        Changed = true;
        Progress = true;
        break;
      }

      // Split: both terminate
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

bool GPUControlFlow::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getSubtarget().getInstrInfo();
  bool Changed = false;

  // Phase 1: Process loops (WHILE/BREAK/JUMP/JOIN)
  MachineLoopInfo &MLI =
      getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  for (MachineLoop *L : MLI)
    processLoop(L);

  // Phase 2: Merge preheaders into loop headers
  for (auto &MBB : MF) {
    bool HasLoopCF = false;
    for (auto &MI : MBB)
      if (MI.getOpcode() == GPU::JUMP_INST)
        HasLoopCF = true;
    if (!HasLoopCF)
      mergeLinearChain(MBB);
  }

  // Phase 3: Un-tail-merge HALTs
  Changed |= unTailMergeHALTs(MF);

  // Phase 4: Convert conditionals to GOTO/JOIN
  Changed |= processAllConditionals(MF);

  // Phase 4b: Remove remaining unconditional branch pseudos
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

  // Offset computation deferred to GPUPeephole (runs after
  // instruction-count-changing optimizations).

  return Changed;
}

FunctionPass *llvm::createGPUControlFlowPass() {
  return new GPUControlFlow();
}
