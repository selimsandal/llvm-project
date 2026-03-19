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
};

} // namespace llvm

#endif
