//===-- GPUISelDAGToDAG.cpp - GPU DAG->DAG Pattern Isel ---===//

#include "GPU.h"
#include "GPUISelLowering.h"
#include "GPUSubtarget.h"
#include "GPUTargetMachine.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/IntrinsicsGPU.h"

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

  case ISD::ATOMIC_LOAD_ADD:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_MAX:
  case ISD::ATOMIC_LOAD_MIN:
  case ISD::ATOMIC_LOAD_UMAX:
  case ISD::ATOMIC_LOAD_UMIN:
  case ISD::ATOMIC_SWAP: {
    SDLoc DL(N);
    auto *AN = cast<AtomicSDNode>(N);
    SDValue Addr = AN->getBasePtr();
    SDValue Val = AN->getVal();

    unsigned AtomicOp;
    switch (Opcode) {
    case ISD::ATOMIC_LOAD_ADD:  AtomicOp = 0; break;
    case ISD::ATOMIC_LOAD_XOR:  AtomicOp = 2; break;
    case ISD::ATOMIC_LOAD_OR:   AtomicOp = 3; break;
    case ISD::ATOMIC_LOAD_MAX:  AtomicOp = 4; break;
    case ISD::ATOMIC_LOAD_MIN:  AtomicOp = 5; break;
    case ISD::ATOMIC_LOAD_UMAX: AtomicOp = 6; break;
    case ISD::ATOMIC_LOAD_UMIN: AtomicOp = 7; break;
    case ISD::ATOMIC_SWAP:      AtomicOp = 8; break;
    case ISD::ATOMIC_LOAD_AND:  AtomicOp = 13; break;
    default: llvm_unreachable("Unexpected atomic op");
    }

    SDValue Ops[] = {
      Addr, Val,
      CurDAG->getTargetConstant(0, DL, MVT::i32), // offset = 0
      CurDAG->getTargetConstant(AtomicOp, DL, MVT::i32),
      AN->getChain()
    };
    SDNode *AtomicNode = CurDAG->getMachineNode(
        GPU::ATOMIC, DL, MVT::i32, MVT::Other, Ops);
    ReplaceNode(N, AtomicNode);
    return;
  }

  case ISD::ATOMIC_CMP_SWAP: {
    SDLoc DL(N);
    auto *AN = cast<AtomicSDNode>(N);
    SDValue Addr = AN->getBasePtr();
    SDValue CmpVal = AN->getOperand(2);
    SDValue SwapVal = AN->getOperand(3);

    SDValue Ops[] = {
      Addr, CmpVal, SwapVal,
      CurDAG->getTargetConstant(0, DL, MVT::i32), // offset = 0
      AN->getChain()
    };
    SDNode *CASNode = CurDAG->getMachineNode(
        GPU::ATOMIC_CAS, DL, MVT::i32, MVT::Other, Ops);
    ReplaceNode(N, CASNode);
    return;
  }

  case ISD::INTRINSIC_WO_CHAIN: {
    SDLoc DL(N);
    unsigned IntrID = N->getConstantOperandVal(0);
    unsigned ReduceOp;
    switch (IntrID) {
    default: break; // Fall through to SelectCode
    case Intrinsic::gpu_reduce_add:  ReduceOp = 0; goto do_reduce;
    case Intrinsic::gpu_reduce_and:  ReduceOp = 1; goto do_reduce;
    case Intrinsic::gpu_reduce_or:   ReduceOp = 2; goto do_reduce;
    case Intrinsic::gpu_reduce_xor:  ReduceOp = 3; goto do_reduce;
    case Intrinsic::gpu_reduce_smin: ReduceOp = 4; goto do_reduce;
    case Intrinsic::gpu_reduce_smax: ReduceOp = 5; goto do_reduce;
    case Intrinsic::gpu_reduce_umin: ReduceOp = 6; goto do_reduce;
    case Intrinsic::gpu_reduce_umax: ReduceOp = 7; goto do_reduce;
    case Intrinsic::gpu_reduce_fadd: ReduceOp = 8; goto do_reduce;
    case Intrinsic::gpu_reduce_fmin: ReduceOp = 9; goto do_reduce;
    case Intrinsic::gpu_reduce_fmax: ReduceOp = 10; goto do_reduce;
    do_reduce: {
      SDValue Src = N->getOperand(1);
      SDNode *ReduceNode = CurDAG->getMachineNode(
          GPU::REDUCE, DL, N->getValueType(0),
          {Src, CurDAG->getTargetConstant(ReduceOp, DL, MVT::i32)});
      ReplaceNode(N, ReduceNode);
      return;
    }
    }
    break; // Fall through to SelectCode for non-GPU intrinsics
  }

  case GPUISD::REDUCE: {
    SDLoc DL(N);
    SDValue Src = N->getOperand(0);
    unsigned ReduceOp = N->getConstantOperandVal(1);
    SDNode *ReduceNode = CurDAG->getMachineNode(
        GPU::REDUCE, DL, N->getValueType(0),
        {Src, CurDAG->getTargetConstant(ReduceOp, DL, MVT::i32)});
    ReplaceNode(N, ReduceNode);
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
