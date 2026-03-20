//===-- GPUArgRegCopy.cpp - Copy ABI arg regs to vregs ---------===//
//
// Direct kernel arguments and resource bindings currently enter compiled
// shaders in fixed physical registers r1-r4. Rewriting those uses onto
// vregs before register allocation lets RA treat them like ordinary values
// instead of reusing the physical ABI registers later in the function.
//
//===----------------------------------------------------------------------===//

#include "GPU.h"
#include "GPUInstrInfo.h"
#include "GPURegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-arg-reg-copy"

namespace {

class GPUArgRegCopy : public MachineFunctionPass {
public:
  static char ID;

  GPUArgRegCopy() : MachineFunctionPass(ID) {
    initializeGPUArgRegCopyPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "GPU Arg Register Copy"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // namespace

char GPUArgRegCopy::ID = 0;

INITIALIZE_PASS(GPUArgRegCopy, "gpu-arg-reg-copy",
                "GPU Arg Register Copy", false, false)

bool GPUArgRegCopy::runOnMachineFunction(MachineFunction &MF) {
  static constexpr MCPhysReg ArgRegs[] = {GPU::R1, GPU::R2, GPU::R3, GPU::R4};

  MachineBasicBlock &Entry = MF.front();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  DenseMap<Register, Register> Copies;
  SmallPtrSet<MachineInstr *, 4> CopyMIs;
  auto InsertPt = Entry.begin();
  bool Changed = false;

  while (InsertPt != Entry.end() &&
         (InsertPt->isPHI() || InsertPt->isDebugInstr()))
    ++InsertPt;

  for (MCPhysReg ArgReg : ArgRegs) {
    bool Used = false;

    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isReg() || !MO.isUse())
            continue;
          if (MO.getReg() == ArgReg) {
            Used = true;
            break;
          }
        }
        if (Used)
          break;
      }
      if (Used)
        break;
    }

    if (!Used)
      continue;

    Register VReg = MRI.createVirtualRegister(&GPU::GPRRegClass);
    Entry.addLiveIn(ArgReg);
    MachineInstr *Copy =
        BuildMI(Entry, InsertPt, DebugLoc(), TII.get(TargetOpcode::COPY), VReg)
            .addReg(ArgReg);
    Copies[ArgReg] = VReg;
    CopyMIs.insert(Copy);
    Changed = true;
  }

  if (!Changed)
    return false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (CopyMIs.contains(&MI))
        continue;

      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.isUse())
          continue;
        auto It = Copies.find(MO.getReg());
        if (It == Copies.end())
          continue;
        MO.setReg(It->second);
      }
    }
  }

  return true;
}

FunctionPass *llvm::createGPUArgRegCopyPass() {
  return new GPUArgRegCopy();
}
