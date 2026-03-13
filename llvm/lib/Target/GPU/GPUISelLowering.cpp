//===-- GPUISelLowering.cpp - GPU DAG Lowering -----------===//

#include "GPUISelLowering.h"
#include "GPUSubtarget.h"
#include "MCTargetDesc/GPUMCTargetDesc.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-lower"

#include "GPUGenCallingConv.inc"

GPUTargetLowering::GPUTargetLowering(const TargetMachine &TM,
                                     const GPUSubtarget &STI)
    : TargetLowering(TM, STI), Subtarget(STI) {

  addRegisterClass(MVT::i32, &GPU::GPRRegClass);
  addRegisterClass(MVT::f32, &GPU::GPRRegClass);

  computeRegisterProperties(STI.getRegisterInfo());

  // Promote small integers to i32
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);

  for (MVT VT : MVT::integer_valuetypes()) {
    for (auto ExtType : {ISD::EXTLOAD, ISD::ZEXTLOAD, ISD::SEXTLOAD}) {
      for (auto SmallVT : {MVT::i1, MVT::i8, MVT::i16})
        setLoadExtAction(ExtType, VT, SmallVT, Promote);
    }
  }

  // No division in hardware
  setOperationAction(ISD::SDIV, MVT::i32, Expand);
  setOperationAction(ISD::UDIV, MVT::i32, Expand);
  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);

  setOperationAction(ISD::MULHU, MVT::i32, Expand);
  setOperationAction(ISD::MULHS, MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::ROTR, MVT::i32, Expand);
  setOperationAction(ISD::BSWAP, MVT::i32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  setOperationAction(ISD::CTLZ, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);
  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);

  // Float
  setOperationAction(ISD::FMA, MVT::f32, Legal);
  setOperationAction(ISD::FMINNUM, MVT::f32, Legal);
  setOperationAction(ISD::FMAXNUM, MVT::f32, Legal);
  setOperationAction(ISD::FNEG, MVT::f32, Expand);
  setOperationAction(ISD::FABS, MVT::f32, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);
  setOperationAction(ISD::FSQRT, MVT::f32, Expand);
  setOperationAction(ISD::FPOW, MVT::f32, Expand);
  setOperationAction(ISD::FSIN, MVT::f32, Expand);
  setOperationAction(ISD::FCOS, MVT::f32, Expand);
  setOperationAction(ISD::FREM, MVT::f32, Expand);

  // Control flow
  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f32, Custom);
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::SETCC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT, MVT::f32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);

  // Globals
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);

  // Stack (not supported)
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Expand);
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::FRAMEADDR, MVT::i32, Expand);
  setOperationAction(ISD::RETURNADDR, MVT::i32, Expand);

  // Varargs
  setOperationAction(ISD::VASTART, MVT::Other, Expand);
  setOperationAction(ISD::VAARG, MVT::Other, Expand);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  setMaxAtomicSizeInBitsSupported(32);
  setBooleanContents(ZeroOrOneBooleanContent);
  setMinFunctionAlignment(Align(32));
}

SDValue GPUTargetLowering::LowerOperation(SDValue Op,
                                          SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("Unexpected operation to lower");
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG);
  case ISD::SETCC:
    return LowerSETCC(Op, DAG);
  case ISD::BR_CC:
    return LowerBR_CC(Op, DAG);
  }
}

const char *GPUTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch ((GPUISD::NodeType)Opcode) {
  case GPUISD::FIRST_NUMBER: break;
  case GPUISD::MOVI:    return "GPUISD::MOVI";
  case GPUISD::CMP:     return "GPUISD::CMP";
  case GPUISD::SEL:     return "GPUISD::SEL";
  case GPUISD::BRCOND:  return "GPUISD::BRCOND";
  case GPUISD::HALT:    return "GPUISD::HALT";
  case GPUISD::RETURN:  return "GPUISD::RETURN";
  case GPUISD::WRAPPER: return "GPUISD::WRAPPER";
  }
  return nullptr;
}

SDValue GPUTargetLowering::LowerGlobalAddress(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  int64_t Offset = cast<GlobalAddressSDNode>(Op)->getOffset();
  SDValue Addr = DAG.getTargetGlobalAddress(GV, DL, MVT::i32, Offset);
  return DAG.getNode(GPUISD::WRAPPER, DL, MVT::i32, Addr);
}

static unsigned mapCondCode(ISD::CondCode CC, bool &NeedInvert) {
  NeedInvert = false;
  switch (CC) {
  default: llvm_unreachable("Unsupported condition code");
  case ISD::SETEQ:  return 0;
  case ISD::SETNE:  NeedInvert = true; return 0;
  case ISD::SETLT:  return 1;
  case ISD::SETGE:  NeedInvert = true; return 1;
  case ISD::SETULT: return 2;
  case ISD::SETUGE: NeedInvert = true; return 2;
  case ISD::SETGT:  return 1;
  case ISD::SETLE:  NeedInvert = true; return 1;
  case ISD::SETUGT: return 2;
  case ISD::SETULE: NeedInvert = true; return 2;
  case ISD::SETOEQ: return 3;
  case ISD::SETONE: NeedInvert = true; return 3;
  case ISD::SETOLT: return 4;
  case ISD::SETOGE: NeedInvert = true; return 4;
  case ISD::SETOGT: return 4;
  case ISD::SETOLE: NeedInvert = true; return 4;
  case ISD::SETO:   return 5;
  case ISD::SETUO:  NeedInvert = true; return 5;
  }
}

static bool needsOperandSwap(ISD::CondCode CC) {
  switch (CC) {
  case ISD::SETGT:
  case ISD::SETLE:
  case ISD::SETUGT:
  case ISD::SETULE:
  case ISD::SETOGT:
  case ISD::SETOLE:
    return true;
  default:
    return false;
  }
}

SDValue GPUTargetLowering::LowerSETCC(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();

  bool NeedInvert;
  unsigned HWCond = mapCondCode(CC, NeedInvert);

  if (needsOperandSwap(CC))
    std::swap(LHS, RHS);

  SDValue Cmp = DAG.getNode(GPUISD::CMP, DL, MVT::Glue,
                            DAG.getTargetConstant(HWCond, DL, MVT::i32),
                            LHS, RHS,
                            DAG.getTargetConstant(0, DL, MVT::i32));

  SDValue One = DAG.getConstant(1, DL, MVT::i32);
  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  SDValue TrueVal = NeedInvert ? Zero : One;
  SDValue FalseVal = NeedInvert ? One : Zero;

  return DAG.getNode(GPUISD::SEL, DL, MVT::i32,
                     TrueVal, FalseVal,
                     DAG.getTargetConstant(0, DL, MVT::i32),
                     Cmp);
}

SDValue GPUTargetLowering::LowerSELECT_CC(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue TrueVal = Op.getOperand(2);
  SDValue FalseVal = Op.getOperand(3);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();

  bool NeedInvert;
  unsigned HWCond = mapCondCode(CC, NeedInvert);

  if (needsOperandSwap(CC))
    std::swap(LHS, RHS);

  SDValue Cmp = DAG.getNode(GPUISD::CMP, DL, MVT::Glue,
                            DAG.getTargetConstant(HWCond, DL, MVT::i32),
                            LHS, RHS,
                            DAG.getTargetConstant(0, DL, MVT::i32));

  if (NeedInvert)
    std::swap(TrueVal, FalseVal);

  return DAG.getNode(GPUISD::SEL, DL, Op.getValueType(),
                     TrueVal, FalseVal,
                     DAG.getTargetConstant(0, DL, MVT::i32),
                     Cmp);
}

SDValue GPUTargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);

  bool NeedInvert;
  unsigned HWCond = mapCondCode(CC, NeedInvert);

  if (needsOperandSwap(CC))
    std::swap(LHS, RHS);

  // Emit CMP -> flag register (via glue)
  SDValue Cmp = DAG.getNode(GPUISD::CMP, DL, MVT::Glue,
                            DAG.getTargetConstant(HWCond, DL, MVT::i32),
                            LHS, RHS,
                            DAG.getTargetConstant(0, DL, MVT::i32));

  // Emit conditional branch. invert=1 means the branch target is reached
  // when the flag is FALSE (inverted condition).
  return DAG.getNode(GPUISD::BRCOND, DL, MVT::Other,
                     Chain,
                     DAG.getTargetConstant(NeedInvert ? 1 : 0, DL, MVT::i32),
                     DAG.getTargetConstant(0, DL, MVT::i32), // flag_reg = f0
                     Dest,
                     Cmp);
}

SDValue GPUTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  static const MCPhysReg ArgRegs[] = {GPU::R1, GPU::R2, GPU::R3, GPU::R4};

  for (unsigned i = 0, e = Ins.size(); i < e; ++i) {
    if (i >= 4)
      report_fatal_error("GPU kernels support at most 4 arguments (r1-r4)");
    Register VReg = MRI.createVirtualRegister(&GPU::GPRRegClass);
    MRI.addLiveIn(ArgRegs[i], VReg);
    SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, VReg, MVT::i32);
    InVals.push_back(ArgValue);
  }

  return Chain;
}

SDValue GPUTargetLowering::LowerReturn(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
    SelectionDAG &DAG) const {
  return DAG.getNode(GPUISD::RETURN, DL, MVT::Other, Chain);
}

Register GPUTargetLowering::getRegisterByName(const char *RegName, LLT VT,
                                              const MachineFunction &MF) const {
  Register Reg = StringSwitch<Register>(RegName)
    .Case("r0", GPU::R0)
    .Case("r1", GPU::R1)
    .Case("r2", GPU::R2)
    .Case("r3", GPU::R3)
    .Case("r4", GPU::R4)
    .Case("r5", GPU::R5)
    .Default(Register());
  if (Reg)
    return Reg;
  report_fatal_error("Invalid register name");
}
