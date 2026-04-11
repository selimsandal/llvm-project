//===-- GPUISelLowering.h - GPU DAG Lowering -----*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_GPU_GPUISELLOWERING_H
#define LLVM_LIB_TARGET_GPU_GPUISELLOWERING_H

#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class GPUSubtarget;

namespace GPUISD {
enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  MOVI,       // Move immediate
  GETSR,      // Read raw special register
  CMP,        // Compare (writes flag register via glue)
  SEL,        // Select based on flag register
  BRCOND,     // Conditional branch (flag-based, with invert)
  REDUCE,     // Cross-lane reduction
  HALT,       // Terminate kernel
  RETURN,     // Return (lowered to HALT)
  WRAPPER,    // Address wrapper
};
} // namespace GPUISD

class GPUTargetLowering : public TargetLowering {
  const GPUSubtarget &Subtarget;

public:
  GPUTargetLowering(const TargetMachine &TM, const GPUSubtarget &STI);

  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  const char *getTargetNodeName(unsigned Opcode) const override;

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;

  SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;

  TargetLoweringBase::AtomicExpansionKind
  shouldExpandAtomicRMWInIR(const AtomicRMWInst *AI) const override;

  TargetLoweringBase::AtomicExpansionKind
  shouldExpandAtomicCmpXchgInIR(const AtomicCmpXchgInst *AI) const override;

  Register getRegisterByName(const char *RegName, LLT VT,
                             const MachineFunction &MF) const override;

  bool shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                         Type *Ty) const override {
    return true;
  }

  bool reduceSelectOfFPConstantLoads(EVT CmpOpVT) const override {
    return false;
  }

  // The GPU has only 32-bit loads. Refuse to narrow a full word load down
  // to an i8/i16 extload: there is no hardware instruction to select, and
  // the legalizer's expand path for EXTLOAD hits an assertion on this
  // target. Keeping the load+AND form means ISel sees a plain i32 load
  // followed by an AND with the mask, which both match cleanly.
  bool shouldReduceLoadWidth(
      SDNode *Load, ISD::LoadExtType ExtTy, EVT NewVT,
      std::optional<unsigned> ByteOffset = std::nullopt) const override {
    if (NewVT.isSimple()) {
      MVT SVT = NewVT.getSimpleVT();
      if (SVT == MVT::i1 || SVT == MVT::i8 || SVT == MVT::i16)
        return false;
    }
    return TargetLowering::shouldReduceLoadWidth(Load, ExtTy, NewVT,
                                                 ByteOffset);
  }
};

} // namespace llvm

#endif
