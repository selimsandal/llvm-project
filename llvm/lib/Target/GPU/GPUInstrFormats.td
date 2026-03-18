//===-- GPUInstrFormats.td - GPU Instruction Formats -*- tablegen -*-===//
//
// 128-bit instruction encoding matching make_instr128() in gpu_isa.hpp
// and hardware decode in gpu_compute_engine.sv.
//
// Encoding (4 x 32-bit words, MSB-first):
//   w[3]: [31:24] opcode, [23:22] pred, [21:20] flag_reg, [19:16] cond,
//         [15] sat, [14] imm_en, [13:12] rsvd, [11:6] dst, [5:0] src0
//   w[2]: [31:30] src0_mod, [29:24] src1, [23:22] src1_mod,
//         [21:16] src2, [15:14] src2_mod, [13:0] rsvd
//   w[1]: [31:0] imm32
//   w[0]: [31:0] reserved (zero)
//
//===--------------------------------------------------------------===//

// Base class for all GPU instructions
class GPUInst<dag outs, dag ins, string asmstr, list<dag> pattern>
    : Instruction {
  field bits<128> Inst;
  let Size = 16;   // 128 bits = 16 bytes
  let Namespace = "GPU";

  // Opcode field
  bits<8> opcode;
  let Inst{127-120} = opcode;

  // Predication
  bits<2> pred = 0b00;    // PRED_ALWAYS default
  let Inst{119-118} = pred;

  bits<2> flag_reg = 0b00;
  let Inst{117-116} = flag_reg;

  // Condition code (CMP subop, reduce op, atomic op)
  bits<4> cond = 0b0000;
  let Inst{115-112} = cond;

  // Saturation (reserved)
  bit sat = 0;
  let Inst{111} = sat;

  // Immediate enable
  bit imm_en = 0;
  let Inst{110} = imm_en;

  // Reserved bits [109:108]
  let Inst{109-108} = 0b00;

  // Register fields
  bits<6> dst = 0;
  let Inst{107-102} = dst;

  bits<6> src0 = 0;
  let Inst{101-96} = src0;

  bits<2> src0_mod = 0b00;
  let Inst{95-94} = src0_mod;

  bits<6> src1 = 0;
  let Inst{93-88} = src1;

  bits<2> src1_mod = 0b00;
  let Inst{87-86} = src1_mod;

  bits<6> src2 = 0;
  let Inst{85-80} = src2;

  bits<2> src2_mod = 0b00;
  let Inst{79-78} = src2_mod;

  // Reserved bits [77:64]
  let Inst{77-64} = 0;

  // Immediate value
  bits<32> imm32 = 0;
  let Inst{63-32} = imm32;

  // Reserved word [31:0]
  let Inst{31-0} = 0;

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
}

// ALU format: dst = src0 OP src1
class GPUInstALU<bits<8> op, dag outs, dag ins, string asmstr,
                 list<dag> pattern>
    : GPUInst<outs, ins, asmstr, pattern> {
  let opcode = op;
}

// ALU with immediate: dst = src0 OP imm32
class GPUInstALUImm<bits<8> op, dag outs, dag ins, string asmstr,
                    list<dag> pattern>
    : GPUInst<outs, ins, asmstr, pattern> {
  let opcode = op;
  let imm_en = 1;
}

// Unary format: dst = OP(src0)
class GPUInstUnary<bits<8> op, dag outs, dag ins, string asmstr,
                   list<dag> pattern>
    : GPUInst<outs, ins, asmstr, pattern> {
  let opcode = op;
}

// Memory format: uses imm32 for beat index or byte offset
class GPUInstMem<bits<8> op, dag outs, dag ins, string asmstr,
                 list<dag> pattern>
    : GPUInst<outs, ins, asmstr, pattern> {
  let opcode = op;
  let imm_en = 1;
}

// Control flow format: uses imm32 for branch offset, flag_reg for condition
class GPUInstCF<bits<8> op, dag outs, dag ins, string asmstr,
                list<dag> pattern>
    : GPUInst<outs, ins, asmstr, pattern> {
  let opcode = op;
}

// Compare format: writes flag register — category 3 (0x60)
class GPUInstCmp<dag outs, dag ins, string asmstr, list<dag> pattern>
    : GPUInst<outs, ins, asmstr, pattern> {
  let opcode = 0x60;
}

// Pseudo instruction (not emitted, lowered before encoding)
class GPUPseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
    : GPUInst<outs, ins, asmstr, pattern> {
  let isPseudo = 1;
}
