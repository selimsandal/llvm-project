//===-- GPUISelDAGToDAG.cpp - GPU DAG->DAG Pattern Isel ---===//

#include "GPU.h"
#include "GPUSubtarget.h"
#include "GPUTargetMachine.h"
#include "llvm/CodeGen/SelectionDAGISel.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-isel"

namespace {

class GPUDAGToDAGISel : public SelectionDAGISel {
  const GPUSubtarget *Subtarget;

public:
  GPUDAGToDAGISel() = delete;

  explicit GPUDAGToDAGISel(GPUTargetMachine &TM)
      : SelectionDAGISel(TM) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    Subtarget = &MF.getSubtarget<GPUSubtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

  void Select(SDNode *N) override;

  // TableGen patterns
  #include "GPUGenDAGISel.inc"
};

class GPUDAGToDAGISelLegacy : public SelectionDAGISelLegacy {
public:
  static char ID;

  GPUDAGToDAGISelLegacy() = delete;

  explicit GPUDAGToDAGISelLegacy(GPUTargetMachine &TM, CodeGenOptLevel OL)
      : SelectionDAGISelLegacy(
            ID, std::make_unique<GPUDAGToDAGISel>(TM)) {}
};

char GPUDAGToDAGISelLegacy::ID = 0;

} // anonymous namespace

INITIALIZE_PASS(GPUDAGToDAGISelLegacy, "gpu-isel",
                "GPU DAG->DAG Instruction Selection", false, false)

void GPUDAGToDAGISel::Select(SDNode *N) {
  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return;
  }

  unsigned Opcode = N->getOpcode();

  switch (Opcode) {
  default:
    break;

  case GPUISD::CMP: {
    SDLoc DL(N);
    unsigned HWCond = N->getConstantOperandVal(0);
    SDValue LHS = N->getOperand(1);
    SDValue RHS = N->getOperand(2);
    unsigned FlagReg = N->getConstantOperandVal(3);

    SDNode *CmpNode = CurDAG->getMachineNode(
        GPU::CMPrr, DL, MVT::Glue,
        {LHS, RHS,
         CurDAG->getTargetConstant(HWCond, DL, MVT::i32),
         CurDAG->getTargetConstant(FlagReg, DL, MVT::i32)});
    ReplaceUses(SDValue(N, 0), SDValue(CmpNode, 0));
    CurDAG->RemoveDeadNode(N);
    return;
  }

  case GPUISD::SEL: {
    SDLoc DL(N);
    SDValue TrueVal = N->getOperand(0);
    SDValue FalseVal = N->getOperand(1);
    unsigned FlagReg = N->getConstantOperandVal(2);
    SDValue Glue = N->getOperand(3);

    SDNode *SelNode = CurDAG->getMachineNode(
        GPU::SEL, DL, N->getValueType(0),
        {TrueVal, FalseVal,
         CurDAG->getTargetConstant(FlagReg, DL, MVT::i32),
         Glue});
    ReplaceNode(N, SelNode);
    return;
  }

  case GPUISD::RETURN: {
    SDLoc DL(N);
    SDNode *Halt = CurDAG->getMachineNode(GPU::HALT, DL, MVT::Other,
                                          N->getOperand(0));
    ReplaceNode(N, Halt);
    return;
  }

  case GPUISD::MOVI: {
    SDLoc DL(N);
    uint32_t Val = N->getConstantOperandVal(0);
    SDNode *MoviNode = CurDAG->getMachineNode(
        GPU::MOVI, DL, MVT::i32,
        CurDAG->getTargetConstant(Val, DL, MVT::i32));
    ReplaceNode(N, MoviNode);
    return;
  }

  case ISD::BR: {
    SDLoc DL(N);
    SDValue Chain = N->getOperand(0);
    SDValue Dest = N->getOperand(1);
    CurDAG->SelectNodeTo(N, GPU::GPU_BR, MVT::Other, Dest, Chain);
    return;
  }

  case GPUISD::BRCOND: {
    SDLoc DL(N);
    SDValue Chain = N->getOperand(0);
    unsigned Invert = N->getConstantOperandVal(1);
    unsigned FReg = N->getConstantOperandVal(2);
    SDValue Dest = N->getOperand(3);
    SDValue Glue = N->getOperand(4);

    SDValue Ops[] = {
      CurDAG->getTargetConstant(Invert, DL, MVT::i32),
      CurDAG->getTargetConstant(FReg, DL, MVT::i32),
      Dest,
      Chain,
      Glue
    };

    CurDAG->SelectNodeTo(N, GPU::GPU_BRCOND, MVT::Other, Ops);
    return;
  }
  }

  SelectCode(N);
}

FunctionPass *llvm::createGPUISelDag(GPUTargetMachine &TM,
                                     CodeGenOptLevel OptLevel) {
  return new GPUDAGToDAGISelLegacy(TM, OptLevel);
}
