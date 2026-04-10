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
//   WHILE:            push empty loop frame
//   BREAK(flag, d, t): accumulate into compiler-selected frame depth d,
//                      jump if EM=0
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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

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

  bool getBranchInfo(MachineBasicBlock &MBB, BranchInfo &Info) const;
  void removeBranchPseudos(MachineBasicBlock &MBB);
  void processLoop(MachineLoop *L);
  bool processAllConditionals(MachineFunction &MF);
  Register findScratchGPR(MachineBasicBlock &MBB,
                          Register Avoid0 = Register(),
                          Register Avoid1 = Register()) const;
  Register findScratchGPRInRegion(ArrayRef<MachineBasicBlock *> Blocks,
                                  Register Avoid0 = Register(),
                                  Register Avoid1 = Register()) const;
  bool clobbersFlag(MachineBasicBlock &MBB, unsigned FReg) const;
  bool clobbersFlag(ArrayRef<MachineBasicBlock *> Blocks, unsigned FReg) const;
  MachineBasicBlock *
  findLinearChainMerge(MachineBasicBlock *Head,
                       SmallVectorImpl<MachineBasicBlock *> &Blocks) const;

  void processDiamond(MachineFunction &MF, MachineBasicBlock &CondBB,
                      MachineBasicBlock *TrueBB, MachineBasicBlock *FalseBB,
                      MachineBasicBlock *MergeBB, unsigned FReg);
  void processDiamondChain(MachineFunction &MF, MachineBasicBlock &CondBB,
                           ArrayRef<MachineBasicBlock *> TrueBlocks,
                           ArrayRef<MachineBasicBlock *> FalseBlocks,
                           MachineBasicBlock *MergeBB, unsigned FReg);

  void processTriangleTrue(MachineFunction &MF, MachineBasicBlock &CondBB,
                           MachineBasicBlock *ThenBB,
                           MachineBasicBlock *MergeBB, unsigned FReg);
  void processTriangleTrueChain(MachineFunction &MF,
                                MachineBasicBlock &CondBB,
                                ArrayRef<MachineBasicBlock *> ThenBlocks,
                                MachineBasicBlock *MergeBB,
                                unsigned FReg);

  void processTriangleFalse(MachineFunction &MF, MachineBasicBlock &CondBB,
                            MachineBasicBlock *ThenBB,
                            MachineBasicBlock *MergeBB, unsigned FReg);
  void processTriangleFalseChain(MachineFunction &MF,
                                 MachineBasicBlock &CondBB,
                                 ArrayRef<MachineBasicBlock *> ThenBlocks,
                                 MachineBasicBlock *MergeBB,
                                 unsigned FReg);

  void processSplit(MachineFunction &MF, MachineBasicBlock &CondBB,
                    MachineBasicBlock *TrueBB, MachineBasicBlock *FalseBB,
                    unsigned FReg);
  void processGuardToMerge(MachineFunction &MF, MachineBasicBlock &CondBB,
                           MachineBasicBlock *BodyBB,
                           MachineBasicBlock *MergeBB, unsigned FReg,
                           bool MergeIsTarget, unsigned Invert);
  bool lowerResidualLinearConditional(MachineFunction &MF,
                                      MachineBasicBlock &CondBB);
  bool lowerResidualGuard(MachineFunction &MF, MachineBasicBlock &CondBB);

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

bool GPUControlFlow::getBranchInfo(MachineBasicBlock &MBB, BranchInfo &Info) const {
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

Register GPUControlFlow::findScratchGPR(MachineBasicBlock &MBB,
                                        Register Avoid0,
                                        Register Avoid1) const {
  static const MCPhysReg Candidates[] = {
      GPU::R29, GPU::R28, GPU::R27, GPU::R26, GPU::R25, GPU::R24, GPU::R23,
      GPU::R22, GPU::R21, GPU::R20, GPU::R19, GPU::R18, GPU::R17, GPU::R16,
      GPU::R15, GPU::R14, GPU::R13, GPU::R12, GPU::R11, GPU::R10, GPU::R9,
      GPU::R8,  GPU::R7,  GPU::R6,  GPU::R5,  GPU::R4,  GPU::R3,  GPU::R2,
      GPU::R1,  GPU::R31};
  const TargetRegisterInfo *TRI = MBB.getParent()->getSubtarget().getRegisterInfo();

  for (MCPhysReg Reg : Candidates) {
    bool LiveOut = false;
    if (Reg == Avoid0 || Reg == Avoid1)
      continue;
    for (const auto &LiveOutReg : MBB.liveouts()) {
      if (TRI->regsOverlap(Reg, LiveOutReg.PhysReg)) {
        LiveOut = true;
        break;
      }
    }
    if (!LiveOut)
      return Reg;
  }

  return Register();
}

Register GPUControlFlow::findScratchGPRInRegion(ArrayRef<MachineBasicBlock *> Blocks,
                                                Register Avoid0,
                                                Register Avoid1) const {
  static const MCPhysReg Candidates[] = {
      GPU::R29, GPU::R28, GPU::R27, GPU::R26, GPU::R25, GPU::R24, GPU::R23,
      GPU::R22, GPU::R21, GPU::R20, GPU::R19, GPU::R18, GPU::R17, GPU::R16,
      GPU::R15, GPU::R14, GPU::R13, GPU::R12, GPU::R11, GPU::R10, GPU::R9,
      GPU::R8,  GPU::R7,  GPU::R6,  GPU::R5,  GPU::R4,  GPU::R3,  GPU::R2,
      GPU::R1,  GPU::R31};

  if (Blocks.empty())
    return Register();

  const TargetRegisterInfo *TRI =
      Blocks.front()->getParent()->getSubtarget().getRegisterInfo();

  for (MCPhysReg Reg : Candidates) {
    bool Busy = false;

    if (Reg == Avoid0 || Reg == Avoid1)
      continue;

    for (size_t BlockIndex = 0; BlockIndex < Blocks.size(); ++BlockIndex) {
      MachineBasicBlock *MBB = Blocks[BlockIndex];
      for (const auto &LiveOutReg : MBB->liveouts()) {
        if (TRI->regsOverlap(Reg, LiveOutReg.PhysReg)) {
          Busy = true;
          break;
        }
      }
      if (Busy)
        break;

      bool ScanOperands = (BlockIndex > 0);
      if (ScanOperands) {
        for (const MachineInstr &MI : *MBB) {
          for (const MachineOperand &MO : MI.operands()) {
            if (!MO.isReg())
              continue;
            Register OpReg = MO.getReg();
            if (!OpReg.isPhysical())
              continue;
            if (TRI->regsOverlap(Reg, OpReg)) {
              Busy = true;
              break;
            }
          }
          if (Busy)
            break;
        }
      }
      if (Busy)
        break;
    }

    if (!Busy)
      return Reg;
  }

  return Register();
}

bool GPUControlFlow::clobbersFlag(MachineBasicBlock &MBB, unsigned FReg) const {
  const Register Flag = flagReg(FReg);
  auto ScanBegin = MBB.begin();

  auto IsStructuredCF = [](unsigned Opc) {
    switch (Opc) {
    case GPU::GOTO_INST:
    case GPU::JOIN_INST:
    case GPU::WHILE_INST:
    case GPU::BREAK_INST:
    case GPU::JUMP_INST:
      return true;
    default:
      return false;
    }
  };

  for (auto It = MBB.begin(), End = MBB.end(); It != End; ++It) {
    if (It->isDebugInstr())
      continue;
    if (IsStructuredCF(It->getOpcode()))
      ScanBegin = std::next(It);
  }

  for (auto It = ScanBegin, End = MBB.end(); It != End; ++It) {
    const MachineInstr &MI = *It;
    if (MI.isDebugInstr())
      continue;

    if (MI.getOpcode() != GPU::CMPrr && MI.getOpcode() != GPU::CMPri)
      continue;

    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef())
        continue;

      Register Reg = MO.getReg();
      if (Reg == Flag)
        return true;
    }

    if (MI.getNumOperands()) {
      const MachineOperand &FlagOp = MI.getOperand(MI.getNumOperands() - 1);
      if (FlagOp.isImm() && static_cast<unsigned>(FlagOp.getImm()) == FReg)
        return true;
    }
  }

  return false;
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

    // Clone exit-block instructions before BREAK. The exit block may
    // contain computations (e.g., sky color) that must execute before
    // BREAK freezes register values. Only clone when the exit block has
    // a single predecessor (this block) — shared exit blocks (like the
    // accumulation block) must NOT be cloned.
    if (ExitBB->pred_size() == 1) {
      for (auto &MI : *ExitBB) {
        if (MI.isTerminator() || MI.isDebugInstr())
          continue;
        MachineInstr *Clone = BB->getParent()->CloneMachineInstr(&MI);
        BB->insert(BB->end(), Clone);
      }
    }

    // BREAK: depth and offset are patched later once final GOTO/WHILE nesting
    // is known after flattening.
    BuildMI(*BB, BB->end(), DebugLoc(), TII->get(GPU::BREAK_INST))
        .addReg(flagReg(Info.FReg))
        .addImm(BreakPred)
        .addImm(0)   // depth placeholder
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

      MachineBasicBlock *LatchExitBB =
          TargetInLoop ? LatchBI.Fallthrough : LatchBI.BranchTarget;

      removeBranchPseudos(*Latch);

      // Convert only the leading latch-exit MOV/MOVI assignments to flag-gated
      // SELs. These are typically PHI/copy materializations that should apply
      // only to lanes exiting through the latch. Local temporaries later in the
      // block must stay in place; hoisting them changes the live ranges seen by
      // the flattened mask-stack program and can corrupt values after JOIN.
      //
      // The old code cleared the whole exit block, which dropped real work on
      // mixed exit blocks. We now preserve the block and only hoist the
      // contiguous copy prefix at block entry.
      if (LatchExitBB && LatchExitBB->pred_size() == 1) {
        bool ExitOnFlagTrue = (BreakPred == 1);
        SmallVector<MachineInstr *, 8> ToDel;
        for (auto &MI : *LatchExitBB) {
          if (MI.isDebugInstr())
            continue;
          if (MI.isTerminator())
            break;
          unsigned Opc = MI.getOpcode();
          if (Opc != GPU::MOVI && Opc != GPU::MOV)
            break;
          if (Opc == GPU::MOVI) {
            // MOVI rX, imm → materialize imm in a dead physical register, then
            // select between the exit value and the current destination based
            // on which flag value exits the loop. This pass runs post-RA, so
            // creating virtual registers here is not legal.
            Register DstReg = MI.getOperand(0).getReg();
            int64_t Imm = MI.getOperand(1).getImm();
            Register TmpReg = findScratchGPR(*Latch, DstReg);
            if (!TmpReg)
              report_fatal_error("GPUControlFlow: no scratch GPR available for latch-exit MOVI hoist");
            BuildMI(*Latch, Latch->end(), DebugLoc(), TII->get(GPU::MOVI))
                .addReg(TmpReg, RegState::Define)
                .addImm(Imm);
            MachineInstrBuilder SelMI =
                BuildMI(*Latch, Latch->end(), DebugLoc(), TII->get(GPU::SEL))
                    .addReg(DstReg, RegState::Define);
            if (ExitOnFlagTrue)
              SelMI.addReg(TmpReg).addReg(DstReg);
            else
              SelMI.addReg(DstReg).addReg(TmpReg);
            SelMI.addImm(LatchBI.FReg);
            ToDel.push_back(&MI);
          } else if (Opc == GPU::MOV) {
            // MOV rX, rY → SEL rX, exit_value, keep_value or the opposite,
            // depending on whether flag=1 or flag=0 exits the loop.
            Register DstReg = MI.getOperand(0).getReg();
            Register SrcReg = MI.getOperand(1).getReg();
            MachineInstrBuilder SelMI =
                BuildMI(*Latch, Latch->end(), DebugLoc(), TII->get(GPU::SEL))
                    .addReg(DstReg, RegState::Define);
            if (ExitOnFlagTrue)
              SelMI.addReg(SrcReg).addReg(DstReg);
            else
              SelMI.addReg(DstReg).addReg(SrcReg);
            SelMI.addImm(LatchBI.FReg);
            ToDel.push_back(&MI);
          }
        }
        for (auto *MI : ToDel)
          MI->eraseFromParent();
      }

      // BREAK for exiting lanes. Depth/offset are patched later.
      BuildMI(*Latch, Latch->end(), DebugLoc(), TII->get(GPU::BREAK_INST))
          .addReg(flagReg(LatchBI.FReg))
          .addImm(BreakPred)
          .addImm(0)   // depth placeholder
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
    removeBranchPseudos(*Header);
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

static void ensureChainAfter(MachineFunction &MF, MachineBasicBlock *Anchor,
                             ArrayRef<MachineBasicBlock *> Chain) {
  MachineBasicBlock *Prev = Anchor;
  for (MachineBasicBlock *BB : Chain) {
    ensureAfter(MF, Prev, BB);
    Prev = BB;
  }
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

static bool isPureExitBlock(const MachineBasicBlock &MBB) {
  bool SawHalt = false;
  for (const MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    if (!MI.isTerminator())
      return false;
    if (MI.getOpcode() == GPU::HALT)
      SawHalt = true;
  }
  return SawHalt;
}

static bool dedupSuccessors(MachineBasicBlock &MBB) {
  SmallVector<MachineBasicBlock *, 4> UniqueSuccs;
  bool Changed = false;

  for (MachineBasicBlock *Succ : MBB.successors()) {
    bool Seen = false;
    for (MachineBasicBlock *Existing : UniqueSuccs) {
      if (Existing == Succ) {
        Seen = true;
        break;
      }
    }
    if (!Seen) {
      UniqueSuccs.push_back(Succ);
      continue;
    }
    Changed = true;
  }

  if (!Changed)
    return false;

  while (!MBB.succ_empty())
    MBB.removeSuccessor(MBB.succ_begin());
  for (MachineBasicBlock *Succ : UniqueSuccs)
    MBB.addSuccessor(Succ);

  return true;
}

static bool dedupAllSuccessors(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= dedupSuccessors(MBB);
  return Changed;
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

      bool OtherHasRealWork = false;
      for (const MachineInstr &MI : *OtherSucc) {
        if (MI.isDebugInstr())
          continue;
        if (!MI.isTerminator()) {
          OtherHasRealWork = true;
          break;
        }
      }
      if (OtherHasRealWork) {
        if (OtherSucc->succ_size() == 1) {
          MachineBasicBlock *ExitBB = *OtherSucc->succ_begin();
          if (ExitBB && isPureExitBlock(*ExitBB) &&
              !Pred->isSuccessor(ExitBB)) {
            removeBranchPseudos(*OtherSucc);
            for (const MachineInstr &MI : *ExitBB) {
              if (MI.isDebugInstr())
                continue;
              MachineInstr *Clone = MF.CloneMachineInstr(&MI);
              OtherSucc->insert(OtherSucc->end(), Clone);
            }
            OtherSucc->removeSuccessor(ExitBB);
            Changed = true;
            Progress = true;
            break;
          }
        }
        continue;
      }

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

bool GPUControlFlow::clobbersFlag(ArrayRef<MachineBasicBlock *> Blocks,
                                  unsigned FReg) const {
  for (MachineBasicBlock *BB : Blocks)
    if (clobbersFlag(*BB, FReg))
      return true;
  return false;
}

MachineBasicBlock *GPUControlFlow::findLinearChainMerge(
    MachineBasicBlock *Head,
    SmallVectorImpl<MachineBasicBlock *> &Blocks) const {
  Blocks.clear();
  if (!Head)
    return nullptr;

  SmallPtrSet<MachineBasicBlock *, 16> Seen;
  MachineBasicBlock *Cur = Head;

  while (true) {
    if (!Seen.insert(Cur).second)
      return nullptr;
    Blocks.push_back(Cur);

    if (Cur->succ_size() != 1)
      return nullptr;

    MachineBasicBlock *Next = *Cur->succ_begin();
    if (Next->pred_size() != 1)
      return Next;

    Cur = Next;
  }
}

void GPUControlFlow::mergeLinearChain(MachineBasicBlock &Head) {
  while (Head.succ_size() == 1) {
    MachineBasicBlock *Next = *Head.succ_begin();
    if (Next->pred_size() != 1)
      break;

    removeBranchPseudos(Head);
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
  constexpr unsigned CondEq = 0;

  ensureAfter(MF, &CondBB, TrueBB);
  ensureAfter(MF, TrueBB, FalseBB);
  if (MergeBB)
    ensureAfter(MF, FalseBB, MergeBB);

  removeBranchPseudos(CondBB);
  removeBranchPseudos(*TrueBB);
  removeBranchPseudos(*FalseBB);

  const bool PreserveOuterCond = clobbersFlag(*TrueBB, FReg);
  unsigned SavedFReg = (FReg + 1) & 3;
  Register SavedMaskReg;
  if (PreserveOuterCond) {
    SmallVector<MachineBasicBlock *, 2> Blocks = {&CondBB, TrueBB};
    SavedMaskReg = findScratchGPRInRegion(Blocks);
    if (!SavedMaskReg)
      report_fatal_error(
          "GPUControlFlow: no scratch GPR available to preserve diamond "
          "condition");
    BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::MOVI))
        .addReg(SavedMaskReg, RegState::Define)
        .addImm(1);
    BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::SELi))
        .addReg(SavedMaskReg, RegState::Define)
        .addReg(SavedMaskReg)
        .addImm(0)
        .addImm(FReg);
  }

  // GOTO: flag=1 lanes (false-path) get pushed, true-path lanes stay active
  // BRCOND flag=1 branches to TrueBB. For GOTO, we push the "other" lanes.
  // Since TrueBB is the flag=1 path, we want flag=0 lanes pushed → PRED_IF_NOT
  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(FReg))
      .addImm(2)   // PRED_IF_NOT: push flag=0 (false) lanes, true lanes active
      .addImm(0);  // JIP offset placeholder

  // JOIN at end of TrueBB (reactivate false-path lanes)
  BuildMI(*TrueBB, TrueBB->end(), DebugLoc(), TII->get(GPU::JOIN_INST));

  if (PreserveOuterCond) {
    BuildMI(*TrueBB, TrueBB->end(), DebugLoc(), TII->get(GPU::CMPri))
        .addReg(SavedMaskReg)
        .addImm(1)
        .addImm(CondEq)
        .addImm(SavedFReg);
  }

  // GOTO: now push true-path lanes (flag=1), false-path lanes active
  BuildMI(*TrueBB, TrueBB->end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(PreserveOuterCond ? SavedFReg : FReg))
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

void GPUControlFlow::processDiamondChain(
    MachineFunction &MF, MachineBasicBlock &CondBB,
    ArrayRef<MachineBasicBlock *> TrueBlocks,
    ArrayRef<MachineBasicBlock *> FalseBlocks,
    MachineBasicBlock *MergeBB, unsigned FReg) {
  constexpr unsigned CondEq = 0;

  assert(!TrueBlocks.empty() && !FalseBlocks.empty() &&
         "diamond chains require non-empty regions");

  MachineBasicBlock *TrueHead = TrueBlocks.front();
  MachineBasicBlock *TrueTail = TrueBlocks.back();
  MachineBasicBlock *FalseHead = FalseBlocks.front();
  MachineBasicBlock *FalseTail = FalseBlocks.back();

  ensureChainAfter(MF, &CondBB, TrueBlocks);
  ensureChainAfter(MF, TrueTail, FalseBlocks);
  if (MergeBB)
    ensureAfter(MF, FalseTail, MergeBB);

  removeBranchPseudos(CondBB);
  removeBranchPseudos(*TrueTail);
  removeBranchPseudos(*FalseTail);

  const bool PreserveOuterCond = clobbersFlag(TrueBlocks, FReg);
  unsigned SavedFReg = (FReg + 1) & 3;
  Register SavedMaskReg;
  if (PreserveOuterCond) {
    SmallVector<MachineBasicBlock *, 8> Blocks;
    Blocks.push_back(&CondBB);
    for (MachineBasicBlock *BB : TrueBlocks)
      Blocks.push_back(BB);
    SavedMaskReg = findScratchGPRInRegion(Blocks);
    if (!SavedMaskReg)
      report_fatal_error(
          "GPUControlFlow: no scratch GPR available to preserve diamond "
          "condition");
    BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::MOVI))
        .addReg(SavedMaskReg, RegState::Define)
        .addImm(1);
    BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::SELi))
        .addReg(SavedMaskReg, RegState::Define)
        .addReg(SavedMaskReg)
        .addImm(0)
        .addImm(FReg);
  }

  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(FReg))
      .addImm(2)
      .addImm(0);

  BuildMI(*TrueTail, TrueTail->end(), DebugLoc(), TII->get(GPU::JOIN_INST));

  if (PreserveOuterCond) {
    BuildMI(*TrueTail, TrueTail->end(), DebugLoc(), TII->get(GPU::CMPri))
        .addReg(SavedMaskReg)
        .addImm(1)
        .addImm(CondEq)
        .addImm(SavedFReg);
  }

  BuildMI(*TrueTail, TrueTail->end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(PreserveOuterCond ? SavedFReg : FReg))
      .addImm(1)
      .addImm(0);

  BuildMI(*FalseTail, FalseTail->end(), DebugLoc(), TII->get(GPU::JOIN_INST));

  while (!CondBB.succ_empty())
    CondBB.removeSuccessor(CondBB.succ_begin());
  CondBB.addSuccessor(TrueHead);

  while (!TrueTail->succ_empty())
    TrueTail->removeSuccessor(TrueTail->succ_begin());
  TrueTail->addSuccessor(FalseHead);

  while (!FalseTail->succ_empty())
    FalseTail->removeSuccessor(FalseTail->succ_begin());
  if (MergeBB)
    FalseTail->addSuccessor(MergeBB);

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

void GPUControlFlow::processTriangleTrueChain(
    MachineFunction &MF, MachineBasicBlock &CondBB,
    ArrayRef<MachineBasicBlock *> ThenBlocks, MachineBasicBlock *MergeBB,
    unsigned FReg) {
  assert(!ThenBlocks.empty() && "triangle chain requires non-empty body");

  MachineBasicBlock *ThenHead = ThenBlocks.front();
  MachineBasicBlock *ThenTail = ThenBlocks.back();

  ensureChainAfter(MF, &CondBB, ThenBlocks);
  ensureAfter(MF, ThenTail, MergeBB);

  removeBranchPseudos(CondBB);
  removeBranchPseudos(*ThenTail);

  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(FReg))
      .addImm(2)
      .addImm(0);

  BuildMI(*ThenTail, ThenTail->end(), DebugLoc(), TII->get(GPU::JOIN_INST));

  while (!CondBB.succ_empty())
    CondBB.removeSuccessor(CondBB.succ_begin());
  CondBB.addSuccessor(ThenHead);

  while (!ThenTail->succ_empty())
    ThenTail->removeSuccessor(ThenTail->succ_begin());
  ThenTail->addSuccessor(MergeBB);

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

void GPUControlFlow::processTriangleFalseChain(
    MachineFunction &MF, MachineBasicBlock &CondBB,
    ArrayRef<MachineBasicBlock *> ThenBlocks, MachineBasicBlock *MergeBB,
    unsigned FReg) {
  assert(!ThenBlocks.empty() && "triangle chain requires non-empty body");

  MachineBasicBlock *ThenHead = ThenBlocks.front();
  MachineBasicBlock *ThenTail = ThenBlocks.back();

  ensureChainAfter(MF, &CondBB, ThenBlocks);
  ensureAfter(MF, ThenTail, MergeBB);

  removeBranchPseudos(CondBB);
  removeBranchPseudos(*ThenTail);

  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(FReg))
      .addImm(1)
      .addImm(0);

  BuildMI(*ThenTail, ThenTail->end(), DebugLoc(), TII->get(GPU::JOIN_INST));

  while (!CondBB.succ_empty())
    CondBB.removeSuccessor(CondBB.succ_begin());
  CondBB.addSuccessor(ThenHead);

  while (!ThenTail->succ_empty())
    ThenTail->removeSuccessor(ThenTail->succ_begin());
  ThenTail->addSuccessor(MergeBB);

  mergeLinearChain(CondBB);
}

// Split: both paths terminate
void GPUControlFlow::processSplit(MachineFunction &MF,
                                  MachineBasicBlock &CondBB,
                                  MachineBasicBlock *TrueBB,
                                  MachineBasicBlock *FalseBB,
                                  unsigned FReg) {
  constexpr unsigned CondEq = 0;

  ensureAfter(MF, &CondBB, TrueBB);
  ensureAfter(MF, TrueBB, FalseBB);

  removeHALTs(*TrueBB);
  removeHALTs(*FalseBB);
  removeBranchPseudos(CondBB);
  removeBranchPseudos(*TrueBB);
  removeBranchPseudos(*FalseBB);

  const bool PreserveOuterCond = clobbersFlag(*TrueBB, FReg);
  unsigned SavedFReg = (FReg + 1) & 3;
  Register SavedMaskReg;
  if (PreserveOuterCond) {
    SmallVector<MachineBasicBlock *, 2> Blocks = {&CondBB, TrueBB};
    SavedMaskReg = findScratchGPRInRegion(Blocks);
    if (!SavedMaskReg)
      report_fatal_error(
          "GPUControlFlow: no scratch GPR available to preserve split "
          "condition");
    BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::MOVI))
        .addReg(SavedMaskReg, RegState::Define)
        .addImm(1);
    BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::SELi))
        .addReg(SavedMaskReg, RegState::Define)
        .addReg(SavedMaskReg)
        .addImm(0)
        .addImm(FReg);
  }

  // Same as diamond but no merge point
  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(FReg))
      .addImm(2)   // PRED_IF_NOT
      .addImm(0);

  BuildMI(*TrueBB, TrueBB->end(), DebugLoc(), TII->get(GPU::JOIN_INST));

  if (PreserveOuterCond) {
    BuildMI(*TrueBB, TrueBB->end(), DebugLoc(), TII->get(GPU::CMPri))
        .addReg(SavedMaskReg)
        .addImm(1)
        .addImm(CondEq)
        .addImm(SavedFReg);
  }

  BuildMI(*TrueBB, TrueBB->end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(PreserveOuterCond ? SavedFReg : FReg))
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

void GPUControlFlow::processGuardToMerge(MachineFunction &MF,
                                         MachineBasicBlock &CondBB,
                                         MachineBasicBlock *BodyBB,
                                         MachineBasicBlock *MergeBB,
                                         unsigned FReg,
                                         bool MergeIsTarget,
                                         unsigned Invert) {
  ensureAfter(MF, &CondBB, BodyBB);

  removeBranchPseudos(CondBB);

  BuildMI(CondBB, CondBB.end(), DebugLoc(), TII->get(GPU::GOTO_INST))
      .addReg(flagReg(FReg))
      .addImm(computePredMode(MergeIsTarget, Invert))
      .addImm(0);

  auto InsertPos = MergeBB->begin();
  while (InsertPos != MergeBB->end() &&
         InsertPos->getOpcode() == GPU::JOIN_INST)
    ++InsertPos;
  BuildMI(*MergeBB, InsertPos, DebugLoc(), TII->get(GPU::JOIN_INST));

  while (!CondBB.succ_empty())
    CondBB.removeSuccessor(CondBB.succ_begin());
  CondBB.addSuccessor(BodyBB);
}

bool GPUControlFlow::lowerResidualGuard(MachineFunction &MF,
                                        MachineBasicBlock &CondBB) {
  BranchInfo Info;
  if (!getBranchInfo(CondBB, Info))
    return false;

  MachineBasicBlock *Next = CondBB.getNextNode();
  if (!Next)
    return false;

  MachineBasicBlock *BodyBB = nullptr;
  MachineBasicBlock *MergeBB = nullptr;
  bool MergeIsTarget = false;

  if (Info.Fallthrough == Next && Info.BranchTarget != Next) {
    BodyBB = Info.Fallthrough;
    MergeBB = Info.BranchTarget;
    MergeIsTarget = true;
  } else if (Info.BranchTarget == Next && Info.Fallthrough != Next) {
    BodyBB = Info.BranchTarget;
    MergeBB = Info.Fallthrough;
    MergeIsTarget = false;
  } else {
    return false;
  }

  bool MergeAfterBody = false;
  for (MachineBasicBlock *BB = BodyBB; BB; BB = BB->getNextNode()) {
    if (BB == MergeBB) {
      MergeAfterBody = true;
      break;
    }
  }
  if (!MergeAfterBody)
    return false;

  processGuardToMerge(MF, CondBB, BodyBB, MergeBB, Info.FReg,
                      MergeIsTarget, Info.Invert);
  return true;
}

bool GPUControlFlow::lowerResidualLinearConditional(
    MachineFunction &MF, MachineBasicBlock &CondBB) {
  BranchInfo Info;
  if (!getBranchInfo(CondBB, Info))
    return false;

  MachineBasicBlock *TruePath =
      (Info.Invert == 0) ? Info.BranchTarget : Info.Fallthrough;
  MachineBasicBlock *FalsePath =
      (Info.Invert == 0) ? Info.Fallthrough : Info.BranchTarget;

  SmallVector<MachineBasicBlock *, 16> TrueBlocks;
  SmallVector<MachineBasicBlock *, 16> FalseBlocks;
  MachineBasicBlock *TrueMerge = findLinearChainMerge(TruePath, TrueBlocks);
  MachineBasicBlock *FalseMerge = findLinearChainMerge(FalsePath, FalseBlocks);

  if (TrueMerge && TrueMerge == FalsePath) {
    processTriangleTrueChain(MF, CondBB, TrueBlocks, FalsePath, Info.FReg);
    return true;
  }

  if (FalseMerge && FalseMerge == TruePath) {
    processTriangleFalseChain(MF, CondBB, FalseBlocks, TruePath, Info.FReg);
    return true;
  }

  if (TrueMerge && FalseMerge && TrueMerge == FalseMerge &&
      TrueMerge != TruePath && TrueMerge != FalsePath) {
    processDiamondChain(MF, CondBB, TrueBlocks, FalseBlocks, TrueMerge,
                        Info.FReg);
    return true;
  }

  return false;
}

bool GPUControlFlow::processAllConditionals(MachineFunction &MF) {
  bool Changed = false;
  auto scanConditionals = [&](bool Reverse) -> bool {
    SmallVector<MachineBasicBlock *, 32> Order;
    Order.reserve(MF.size());

    if (Reverse) {
      for (auto It = MF.rbegin(), E = MF.rend(); It != E; ++It)
        Order.push_back(&*It);
    } else {
      for (auto &MBB : MF)
        Order.push_back(&MBB);
    }

    for (MachineBasicBlock *MBBPtr : Order) {
      MachineBasicBlock &MBB = *MBBPtr;
      BranchInfo Info;
      if (!getBranchInfo(MBB, Info))
        continue;

      MachineBasicBlock *TruePath =
          (Info.Invert == 0) ? Info.BranchTarget : Info.Fallthrough;
      MachineBasicBlock *FalsePath =
          (Info.Invert == 0) ? Info.Fallthrough : Info.BranchTarget;

      // Triangle (true body)
      if (TruePath->succ_size() == 1 &&
          *TruePath->succ_begin() == FalsePath) {
        processTriangleTrue(MF, MBB, TruePath, FalsePath, Info.FReg);
        Changed = true;
        return true;
      }

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
          return true;
        }

        processDiamond(MF, MBB, TruePath, FalsePath, MergeBB, Info.FReg);
        Changed = true;
        return true;
      }

      // Triangle (false body)
      if (FalsePath->succ_size() == 1 &&
          *FalsePath->succ_begin() == TruePath) {
        processTriangleFalse(MF, MBB, FalsePath, TruePath, Info.FReg);
        Changed = true;
        return true;
      }

      // Split: both terminate
      if (TruePath->succ_size() == 0 && FalsePath->succ_size() == 0) {
        processSplit(MF, MBB, TruePath, FalsePath, Info.FReg);
        Changed = true;
        return true;
      }

      // Guard: one path exits the function, the other continues through the
      // rest of the body until a shared final exit block. Park the exit lanes
      // with a GOTO and reactivate them at the exit block with JOIN.
      if (TruePath->succ_size() == 0 && FalsePath->succ_size() != 0) {
        processGuardToMerge(MF, MBB, FalsePath, TruePath, Info.FReg,
                            /*MergeIsTarget=*/Info.BranchTarget == TruePath,
                            Info.Invert);
        Changed = true;
        return true;
      }
      if (FalsePath->succ_size() == 0 && TruePath->succ_size() != 0) {
        processGuardToMerge(MF, MBB, TruePath, FalsePath, Info.FReg,
                            /*MergeIsTarget=*/Info.BranchTarget == FalsePath,
                            Info.Invert);
        Changed = true;
        return true;
      }
    }

    return false;
  };

  while (scanConditionals(/*Reverse=*/true) ||
         scanConditionals(/*Reverse=*/false)) {
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

  // Phase 0: Make all fallthroughs explicit with GPU_BR.
  // LLVM's block placement may put blocks in unexpected order. Without
  // explicit branches, getBranchInfo derives fallthroughs from layout
  // order, which can be wrong after block placement reorders blocks.
  for (auto &MBB : MF) {
    if (MBB.succ_empty()) continue;
    // Check if block ends with an explicit terminator for all paths
    bool HasBR = false, HasHALT = false;
    for (auto It = MBB.end(); It != MBB.begin();) {
      --It;
      if (It->isDebugInstr()) continue;
      if (!It->isTerminator()) break;
      if (It->getOpcode() == GPU::GPU_BR) HasBR = true;
      if (It->getOpcode() == GPU::HALT) HasHALT = true;
    }
    if (HasHALT || HasBR) continue;
    // Block has BRCOND but no explicit fallthrough, or no terminator at all.
    // Add GPU_BR for the fallthrough successor.
    MachineBasicBlock *Fallthrough = MBB.getNextNode();
    if (Fallthrough && MBB.isSuccessor(Fallthrough)) {
      BuildMI(MBB, MBB.end(), DebugLoc(), TII->get(GPU::GPU_BR))
          .addMBB(Fallthrough);
    }
  }
  Changed |= dedupAllSuccessors(MF);

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
  Changed |= dedupAllSuccessors(MF);

  // Phase 4: Convert conditionals to GOTO/JOIN
  Changed |= processAllConditionals(MF);
  Changed |= dedupAllSuccessors(MF);

  // Phase 4b: Remove remaining unconditional branch pseudos anywhere in the
  // block. After loop/conditional lowering they are only CFG bookkeeping, and
  // merged blocks can otherwise leave stale GPU_BR instructions stranded in
  // the middle of a block.
  bool LoweredResidualGuard = true;
  while (LoweredResidualGuard) {
    LoweredResidualGuard = false;
    for (auto &MBB : MF) {
      if (lowerResidualLinearConditional(MF, MBB) ||
          lowerResidualGuard(MF, MBB)) {
        Changed = true;
        LoweredResidualGuard = true;
        break;
      }
    }
    if (LoweredResidualGuard)
      Changed |= dedupAllSuccessors(MF);
  }

  for (auto &MBB : MF) {
    SmallVector<MachineInstr *, 4> ToErase;
    for (auto &MI : MBB) {
      if (MI.getOpcode() == GPU::GPU_BR)
        ToErase.push_back(&MI);
      if (MI.getOpcode() == GPU::GPU_BRCOND) {
        errs() << "GPUControlFlow residual GPU_BRCOND in bb." << MBB.getNumber()
               << ": ";
        MI.print(errs());
        errs() << '\n';
        report_fatal_error("GPUControlFlow: unlowered GPU_BRCOND survived conditional lowering");
      }
    }
    for (auto *MI : ToErase)
      MI->eraseFromParent();
    Changed |= !ToErase.empty();
  }

  // Phase 5a: Reorder blocks to reverse post-order (RPO).
  // LLVM's block placement may put loop-internal blocks after the latch.
  // RPO ensures loop bodies come before loop exits, and all predecessors
  // come before successors (except back-edges).
  {
    SmallVector<MachineBasicBlock *, 32> RPO;
    SmallPtrSet<MachineBasicBlock *, 32> Visited;
    SmallVector<MachineBasicBlock *, 32> PostOrder;

    // Iterative post-order DFS
    SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock::succ_iterator>, 16> Stack;
    MachineBasicBlock *Entry = &MF.front();
    Visited.insert(Entry);
    Stack.push_back({Entry, Entry->succ_begin()});
    while (!Stack.empty()) {
      auto &[BB, It] = Stack.back();
      if (It == BB->succ_end()) {
        PostOrder.push_back(BB);
        Stack.pop_back();
      } else {
        MachineBasicBlock *Succ = *It++;
        if (Visited.insert(Succ).second)
          Stack.push_back({Succ, Succ->succ_begin()});
      }
    }
    // Reverse for RPO
    for (auto It = PostOrder.rbegin(); It != PostOrder.rend(); ++It)
      RPO.push_back(*It);
    // Add unreachable blocks
    for (auto &MBB : MF)
      if (Visited.insert(&MBB).second)
        RPO.push_back(&MBB);
    // Reorder MachineFunction
    for (auto *BB : RPO)
      MF.splice(MF.end(), BB);
  }

  // Phase 5b: Merge all MBBs into a single flat block
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
