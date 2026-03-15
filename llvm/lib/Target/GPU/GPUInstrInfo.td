//===-- GPUInstrInfo.td - GPU Instruction defs ---*- tablegen -*-===//
//
// All instruction definitions for the custom GPU ISA.
// Maps LLVM ISD nodes to GPU hardware opcodes.
//
//===--------------------------------------------------------------===//

include "GPUInstrFormats.td"

//===--------------------------------------------------------------===//
// Operand definitions
//===--------------------------------------------------------------===//

def GPUimm32 : Operand<i32> {
  let PrintMethod = "printImm32Operand";
}

def GPUimm32f : Operand<f32> {
  let PrintMethod = "printImm32Operand";
}

def brtarget : Operand<OtherVT> {
  let PrintMethod = "printBrTargetOperand";
}

// Condition code operand for CMP
def GPUCondCode : Operand<i32> {
  let PrintMethod = "printCondCodeOperand";
}

//===--------------------------------------------------------------===//
// SDNode definitions for GPU-specific nodes
//===--------------------------------------------------------------===//

def GPUcmp     : SDNode<"GPUISD::CMP",
                   SDTypeProfile<0, 4, [SDTCisVT<0, i32>,
                                        SDTCisVT<1, i32>,
                                        SDTCisVT<2, i32>,
                                        SDTCisVT<3, i32>]>,
                   [SDNPOutGlue]>;

def GPUcmpf    : SDNode<"GPUISD::CMP",
                   SDTypeProfile<0, 4, [SDTCisVT<0, i32>,
                                        SDTCisVT<1, f32>,
                                        SDTCisVT<2, f32>,
                                        SDTCisVT<3, i32>]>,
                   [SDNPOutGlue]>;

def GPUsel     : SDNode<"GPUISD::SEL",
                   SDTypeProfile<1, 3, [SDTCisVT<0, i32>,
                                        SDTCisVT<1, i32>,
                                        SDTCisVT<2, i32>,
                                        SDTCisVT<3, i32>]>,
                   [SDNPInGlue]>;

def GPUself    : SDNode<"GPUISD::SEL",
                   SDTypeProfile<1, 3, [SDTCisVT<0, f32>,
                                        SDTCisVT<1, f32>,
                                        SDTCisVT<2, f32>,
                                        SDTCisVT<3, i32>]>,
                   [SDNPInGlue]>;

def GPUmovi    : SDNode<"GPUISD::MOVI",
                   SDTypeProfile<1, 1, [SDTCisVT<0, i32>,
                                        SDTCisVT<1, i32>]>>;

def GPUhalt    : SDNode<"GPUISD::HALT",
                   SDTypeProfile<0, 0, []>,
                   [SDNPHasChain]>;

def GPUreturn  : SDNode<"GPUISD::RETURN",
                   SDTypeProfile<0, 0, []>,
                   [SDNPHasChain]>;

def GPUwrapper : SDNode<"GPUISD::WRAPPER",
                   SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>,
                                        SDTCisPtrTy<0>]>>;

//===--------------------------------------------------------------===//
// Integer ALU Instructions (0x0_)
//===--------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

def NOP : GPUInst<(outs), (ins), "nop", []> {
  let opcode = 0x00;
}

def MOV : GPUInstUnary<0x01,
  (outs GPR:$dst), (ins GPR:$src0),
  "mov\t$dst, $src0", []>;

def MOVI : GPUInstALUImm<0x02,
  (outs GPR:$dst), (ins GPUimm32:$imm32),
  "movi\t$dst, $imm32",
  [(set GPR:$dst, (GPUmovi imm:$imm32))]> {
  let src0 = 0;
}

def ADD : GPUInstALU<0x03,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "add\t$dst, $src0, $src1",
  [(set GPR:$dst, (add GPR:$src0, GPR:$src1))]>;

def ADDi : GPUInstALUImm<0x03,
  (outs GPR:$dst), (ins GPR:$src0, GPUimm32:$imm32),
  "add\t$dst, $src0, $imm32",
  [(set GPR:$dst, (add GPR:$src0, imm:$imm32))]>;

def SUB : GPUInstALU<0x04,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "sub\t$dst, $src0, $src1",
  [(set GPR:$dst, (sub GPR:$src0, GPR:$src1))]>;

def SUBi : GPUInstALUImm<0x04,
  (outs GPR:$dst), (ins GPR:$src0, GPUimm32:$imm32),
  "sub\t$dst, $src0, $imm32",
  [(set GPR:$dst, (sub GPR:$src0, imm:$imm32))]>;

def MUL : GPUInstALU<0x05,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "mul\t$dst, $src0, $src1",
  [(set GPR:$dst, (mul GPR:$src0, GPR:$src1))]>;

def MULi : GPUInstALUImm<0x05,
  (outs GPR:$dst), (ins GPR:$src0, GPUimm32:$imm32),
  "mul\t$dst, $src0, $imm32",
  [(set GPR:$dst, (mul GPR:$src0, imm:$imm32))]>;

def AND : GPUInstALU<0x06,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "and\t$dst, $src0, $src1",
  [(set GPR:$dst, (and GPR:$src0, GPR:$src1))]>;

def ANDi : GPUInstALUImm<0x06,
  (outs GPR:$dst), (ins GPR:$src0, GPUimm32:$imm32),
  "and\t$dst, $src0, $imm32",
  [(set GPR:$dst, (and GPR:$src0, imm:$imm32))]>;

def OR : GPUInstALU<0x07,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "or\t$dst, $src0, $src1",
  [(set GPR:$dst, (or GPR:$src0, GPR:$src1))]>;

def ORi : GPUInstALUImm<0x07,
  (outs GPR:$dst), (ins GPR:$src0, GPUimm32:$imm32),
  "or\t$dst, $src0, $imm32",
  [(set GPR:$dst, (or GPR:$src0, imm:$imm32))]>;

def XOR : GPUInstALU<0x08,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "xor\t$dst, $src0, $src1",
  [(set GPR:$dst, (xor GPR:$src0, GPR:$src1))]>;

def XORi : GPUInstALUImm<0x08,
  (outs GPR:$dst), (ins GPR:$src0, GPUimm32:$imm32),
  "xor\t$dst, $src0, $imm32",
  [(set GPR:$dst, (xor GPR:$src0, imm:$imm32))]>;

def NOT : GPUInstUnary<0x09,
  (outs GPR:$dst), (ins GPR:$src0),
  "not\t$dst, $src0",
  [(set GPR:$dst, (not GPR:$src0))]>;

def SHL : GPUInstALU<0x0A,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "shl\t$dst, $src0, $src1",
  [(set GPR:$dst, (shl GPR:$src0, GPR:$src1))]>;

def SHLi : GPUInstALUImm<0x0A,
  (outs GPR:$dst), (ins GPR:$src0, GPUimm32:$imm32),
  "shl\t$dst, $src0, $imm32",
  [(set GPR:$dst, (shl GPR:$src0, imm:$imm32))]>;

def SHR : GPUInstALU<0x0B,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "shr\t$dst, $src0, $src1",
  [(set GPR:$dst, (srl GPR:$src0, GPR:$src1))]>;

def SHRi : GPUInstALUImm<0x0B,
  (outs GPR:$dst), (ins GPR:$src0, GPUimm32:$imm32),
  "shr\t$dst, $src0, $imm32",
  [(set GPR:$dst, (srl GPR:$src0, imm:$imm32))]>;

def SHRA : GPUInstALU<0x0C,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "shra\t$dst, $src0, $src1",
  [(set GPR:$dst, (sra GPR:$src0, GPR:$src1))]>;

def SHRAi : GPUInstALUImm<0x0C,
  (outs GPR:$dst), (ins GPR:$src0, GPUimm32:$imm32),
  "shra\t$dst, $src0, $imm32",
  [(set GPR:$dst, (sra GPR:$src0, imm:$imm32))]>;

def NEG : GPUInstUnary<0x0D,
  (outs GPR:$dst), (ins GPR:$src0),
  "neg\t$dst, $src0",
  [(set GPR:$dst, (ineg GPR:$src0))]>;

// Cross-lane reduction: dst = reduce(src0, cond)
// Result broadcast to all active lanes. hasSideEffects because convergent.
let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
def REDUCE : GPUInst<(outs GPR:$dst),
  (ins GPR:$src0, i32imm:$reduce_op),
  "reduce\t$dst, $src0", []> {
  let opcode = 0x0F;
}
}

// SEL uses flag_reg field, not a GPR for condition
def SEL : GPUInst<(outs GPR:$dst), (ins GPR:$src0, GPR:$src1, i32imm:$freg),
  "sel\t$dst, $src0, $src1", []> {
  let opcode = 0x0E;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let Uses = [F0]; // Reads flag register written by preceding CMP
}

// Integer min/max — native single-cycle ALU instructions
def SMINrr : GPUInstALU<0x16,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "smin\t$dst, $src0, $src1",
  [(set GPR:$dst, (smin GPR:$src0, GPR:$src1))]>;
def SMAXrr : GPUInstALU<0x17,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "smax\t$dst, $src0, $src1",
  [(set GPR:$dst, (smax GPR:$src0, GPR:$src1))]>;
def UMINrr : GPUInstALU<0x18,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "umin\t$dst, $src0, $src1",
  [(set GPR:$dst, (umin GPR:$src0, GPR:$src1))]>;
def UMAXrr : GPUInstALU<0x19,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "umax\t$dst, $src0, $src1",
  [(set GPR:$dst, (umax GPR:$src0, GPR:$src1))]>;

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

//===--------------------------------------------------------------===//
// Memory Instructions (0x1_)
//===--------------------------------------------------------------===//

let hasSideEffects = 0 in {

let mayLoad = 1, mayStore = 0 in {
def LDV : GPUInstMem<0x10,
  (outs GPR:$dst), (ins GPUimm32:$imm32),
  "ldv\t$dst, $imm32", []>;

def LD_SCATTER : GPUInstMem<0x12,
  (outs GPR:$dst), (ins GPR:$src0, GPUimm32:$imm32),
  "ld_scatter\t$dst, [$src0 + $imm32]",
  [(set i32:$dst, (load (add i32:$src0, imm:$imm32)))]>;
}

let mayLoad = 0, mayStore = 1 in {
def STV : GPUInstMem<0x11,
  (outs), (ins GPR:$src0, GPUimm32:$imm32),
  "stv\t$src0, $imm32", []>;

def ST_SCATTER : GPUInstMem<0x13,
  (outs), (ins GPR:$src0, GPR:$src1, GPUimm32:$imm32),
  "st_scatter\t[$src0 + $imm32], $src1",
  [(store i32:$src1, (add i32:$src0, imm:$imm32))]>;
}

let mayLoad = 1, mayStore = 1, hasSideEffects = 1 in {
def ATOMIC : GPUInst<(outs GPR:$dst),
  (ins GPR:$src0, GPR:$src1, GPUimm32:$imm32, i32imm:$atomic_op),
  "atomic\t$dst, [$src0 + $imm32], $src1", []> {
  let opcode = 0x14;
  let imm_en = 1;
}

// CAS needs 3 GPR sources: addr, compare, swap
def ATOMIC_CAS : GPUInst<(outs GPR:$dst),
  (ins GPR:$src0, GPR:$src1, GPR:$src2, GPUimm32:$imm32),
  "atomic_cas\t$dst, [$src0 + $imm32], $src1, $src2", []> {
  let opcode = 0x14;
  let imm_en = 1;
  let cond = 9;
}
}

// Pixel output (ROP) - side effects
let mayLoad = 0, mayStore = 1, hasSideEffects = 1 in {
def PIXEL_OUT : GPUInst<(outs),
  (ins GPR:$src0, GPR:$src1, GPR:$src2),
  "pixel_out\t$src0, $src1, $src2", []> {
  let opcode = 0x15;
  let dst = 31;
}
}

} // hasSideEffects = 0

//===--------------------------------------------------------------===//
// Float ALU Instructions (0x2_)
//===--------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

def FADD : GPUInstALU<0x20,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "fadd\t$dst, $src0, $src1",
  [(set GPR:$dst, (fadd GPR:$src0, GPR:$src1))]>;

def FADDi : GPUInstALUImm<0x20,
  (outs GPR:$dst), (ins GPR:$src0, GPUimm32:$imm32),
  "fadd\t$dst, $src0, $imm32", []>;

def FMUL : GPUInstALU<0x21,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "fmul\t$dst, $src0, $src1",
  [(set GPR:$dst, (fmul GPR:$src0, GPR:$src1))]>;

def FMULi : GPUInstALUImm<0x21,
  (outs GPR:$dst), (ins GPR:$src0, GPUimm32:$imm32),
  "fmul\t$dst, $src0, $imm32", []>;

def FSUB : GPUInstALU<0x22,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "fsub\t$dst, $src0, $src1",
  [(set GPR:$dst, (fsub GPR:$src0, GPR:$src1))]>;

def FSUBi : GPUInstALUImm<0x22,
  (outs GPR:$dst), (ins GPR:$src0, GPUimm32:$imm32),
  "fsub\t$dst, $src0, $imm32", []>;

def FDIV : GPUInstALU<0x23,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "fdiv\t$dst, $src0, $src1",
  [(set GPR:$dst, (fdiv GPR:$src0, GPR:$src1))]>;

def FMIN : GPUInstALU<0x24,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "fmin\t$dst, $src0, $src1",
  [(set GPR:$dst, (fminnum GPR:$src0, GPR:$src1))]>;

def FMAX : GPUInstALU<0x25,
  (outs GPR:$dst), (ins GPR:$src0, GPR:$src1),
  "fmax\t$dst, $src0, $src1",
  [(set GPR:$dst, (fmaxnum GPR:$src0, GPR:$src1))]>;

def ITOF : GPUInstUnary<0x26,
  (outs GPR:$dst), (ins GPR:$src0),
  "itof\t$dst, $src0",
  [(set GPR:$dst, (sint_to_fp GPR:$src0))]>;

def FTOI : GPUInstUnary<0x27,
  (outs GPR:$dst), (ins GPR:$src0),
  "ftoi\t$dst, $src0",
  [(set GPR:$dst, (fp_to_sint GPR:$src0))]>;

def FMA : GPUInst<(outs GPR:$dst), (ins GPR:$src0, GPR:$src1, GPR:$src2),
  "fma\t$dst, $src0, $src1, $src2",
  [(set GPR:$dst, (fma GPR:$src0, GPR:$src1, GPR:$src2))]> {
  let opcode = 0x28;
}

def UITOF : GPUInstUnary<0x29,
  (outs GPR:$dst), (ins GPR:$src0),
  "uitof\t$dst, $src0",
  [(set GPR:$dst, (uint_to_fp GPR:$src0))]>;

def FTOU : GPUInstUnary<0x2A,
  (outs GPR:$dst), (ins GPR:$src0),
  "ftou\t$dst, $src0",
  [(set GPR:$dst, (fp_to_uint GPR:$src0))]>;

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

//===--------------------------------------------------------------===//
// Compare Instructions (0x30)
//===--------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

// CMP writes to flag register (implicit def via Glue)
def CMPrr : GPUInstCmp<
  (outs), (ins GPR:$src0, GPR:$src1, i32imm:$cc, i32imm:$freg),
  "cmp\t$src0, $src1", []> {
  let Defs = [F0];
}

def CMPri : GPUInst<
  (outs), (ins GPR:$src0, GPUimm32:$imm32, i32imm:$cc, i32imm:$freg),
  "cmp\t$src0, $imm32", []> {
  let opcode = 0x30;
  let imm_en = 1;
  let Defs = [F0];
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

//===--------------------------------------------------------------===//
// Control Flow Instructions (0x4_)
//===--------------------------------------------------------------===//

let hasSideEffects = 1 in {

// Mask-stack operations — NOT terminators/branches from LLVM's perspective.
// The control flow pass inserts these into the middle of a linearized block.
def IF_INST : GPUInstCF<0x40,
  (outs), (ins i32imm:$freg, GPUimm32:$imm32),
  "if\tf$freg, $imm32", []> {
  let imm_en = 1;
}

def ELSE_INST : GPUInstCF<0x41,
  (outs), (ins GPUimm32:$imm32),
  "else\t$imm32", []> {
  let imm_en = 1;
}

def ENDIF_INST : GPUInstCF<0x42,
  (outs), (ins),
  "endif", []>;

def LOOP_INST : GPUInstCF<0x43,
  (outs), (ins),
  "loop", []>;

def ENDLOOP_INST : GPUInstCF<0x44,
  (outs), (ins i32imm:$freg, GPUimm32:$imm32),
  "endloop\tf$freg, $imm32", []> {
  let imm_en = 1;
}

def BREAK_INST : GPUInstCF<0x45,
  (outs), (ins i32imm:$freg, GPUimm32:$imm32),
  "break\tf$freg, $imm32", []> {
  let imm_en = 1;
}

let isTerminator = 1, isReturn = 1, isBarrier = 1 in {
def HALT : GPUInst<(outs), (ins), "halt",
  [(GPUhalt)]> {
  let opcode = 0xFF;
}
}

} // hasSideEffects = 1

//===--------------------------------------------------------------===//
// Pseudo instructions (lowered before encoding)
//===--------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

// Return pseudo - lowered to HALT
let isTerminator = 1, isReturn = 1, isBarrier = 1 in {
def RET : GPUPseudo<(outs), (ins), "ret", [(GPUreturn)]>;
}

// Select pseudo - expanded in ISelLowering to CMP + SEL sequence
def SELECT_I32 : GPUPseudo<
  (outs GPR:$dst), (ins GPR:$cond, GPR:$t, GPR:$f),
  "select\t$dst, $cond, $t, $f", []>;

def SELECT_F32 : GPUPseudo<
  (outs GPR:$dst), (ins GPR:$cond, GPR:$t, GPR:$f),
  "selectf\t$dst, $cond, $t, $f", []>;

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

// Spill/reload pseudos for register spilling via scatter load/store
let hasSideEffects = 0 in {
let mayLoad = 0, mayStore = 1 in {
def SPILL_GPR : GPUPseudo<
  (outs), (ins GPR:$src, i32imm:$fi),
  "spill\t$src, $fi", []>;
}
let mayLoad = 1, mayStore = 0 in {
def RELOAD_GPR : GPUPseudo<
  (outs GPR:$dst), (ins i32imm:$fi),
  "reload\t$dst, $fi", []>;
}
}

// Conditional branch pseudo (lowered to IF/ELSE/ENDIF by GPUControlFlow pass)
// Uses F0 to keep the CMP that writes F0 alive through dead code elimination.
let isBranch = 1, isTerminator = 1, hasSideEffects = 0,
    mayLoad = 0, mayStore = 0 in {
def GPU_BRCOND : GPUPseudo<
  (outs), (ins i32imm:$invert, i32imm:$freg, brtarget:$target),
  "gpu_brcond\t$invert, f$freg, $target", []> {
  let Uses = [F0];
}
}

// Unconditional branch pseudo (removed by GPUControlFlow pass)
let isBranch = 1, isTerminator = 1, isBarrier = 1, hasSideEffects = 0,
    mayLoad = 0, mayStore = 0 in {
def GPU_BR : GPUPseudo<
  (outs), (ins brtarget:$target),
  "gpu_br\t$target", []>;
}

//===--------------------------------------------------------------===//
// Patterns
//===--------------------------------------------------------------===//

// Constant materialization
def : Pat<(i32 imm:$val), (MOVI imm:$val)>;

// Bitcast between i32 and f32 is free (shared GPR register class)
def : Pat<(f32 (bitconvert i32:$src)), (COPY_TO_REGCLASS GPR:$src, GPR)>;
def : Pat<(i32 (bitconvert f32:$src)), (COPY_TO_REGCLASS GPR:$src, GPR)>;

// f32 constants - handled via custom lowering (bitcast to i32 MOVI)

// Simple load from address (i32 and f32)
def : Pat<(i32 (load i32:$addr)), (LD_SCATTER GPR:$addr, 0)>;
def : Pat<(f32 (load i32:$addr)), (LD_SCATTER GPR:$addr, 0)>;

// f32 load with offset
def : Pat<(f32 (load (add i32:$base, imm:$off))),
          (LD_SCATTER GPR:$base, imm:$off)>;

// Simple store to address (i32 and f32)
def : Pat<(store i32:$val, i32:$addr), (ST_SCATTER GPR:$addr, GPR:$val, 0)>;
def : Pat<(store f32:$val, i32:$addr), (ST_SCATTER GPR:$addr, GPR:$val, 0)>;

// f32 store with offset
def : Pat<(store f32:$val, (add i32:$base, imm:$off)),
          (ST_SCATTER GPR:$base, GPR:$val, imm:$off)>;
