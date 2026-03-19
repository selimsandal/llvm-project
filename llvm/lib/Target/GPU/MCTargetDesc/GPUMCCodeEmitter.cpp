//===-- GPUMCCodeEmitter.cpp - GPU MC Code Emitter --------===//
//
// Encodes MCInst into 128-bit GPU instructions matching make_instr128()
// bit layout from gpu_isa.hpp. Two instructions packed per 256-bit beat.
//
//===--------------------------------------------------------------===//

#include "MCTargetDesc/GPUMCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-mc-code-emitter"

namespace {

class GPUMCCodeEmitter : public MCCodeEmitter {
  const MCInstrInfo &MCII;
  MCContext &Ctx;

public:
  GPUMCCodeEmitter(const MCInstrInfo &MCII, MCContext &Ctx)
      : MCII(MCII), Ctx(Ctx) {}

  void encodeInstruction(const MCInst &MI, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

private:
  // Get the hardware encoding of a register
  unsigned getRegEncoding(unsigned Reg) const;

  // Build the 4 x uint32_t words for a 128-bit instruction
  void encode128(const MCInst &MI, uint32_t W[4]) const;
};

} // anonymous namespace

unsigned GPUMCCodeEmitter::getRegEncoding(unsigned Reg) const {
  switch (Reg) {
  case GPU::R0:  return 0;  case GPU::R1:  return 1;
  case GPU::R2:  return 2;  case GPU::R3:  return 3;
  case GPU::R4:  return 4;  case GPU::R5:  return 5;
  case GPU::R6:  return 6;  case GPU::R7:  return 7;
  case GPU::R8:  return 8;  case GPU::R9:  return 9;
  case GPU::R10: return 10; case GPU::R11: return 11;
  case GPU::R12: return 12; case GPU::R13: return 13;
  case GPU::R14: return 14; case GPU::R15: return 15;
  case GPU::R16: return 16; case GPU::R17: return 17;
  case GPU::R18: return 18; case GPU::R19: return 19;
  case GPU::R20: return 20; case GPU::R21: return 21;
  case GPU::R22: return 22; case GPU::R23: return 23;
  case GPU::R24: return 24; case GPU::R25: return 25;
  case GPU::R26: return 26; case GPU::R27: return 27;
  case GPU::R28: return 28; case GPU::R29: return 29;
  case GPU::R30: return 30; case GPU::R31: return 31;
  case GPU::F0:  return 0;  case GPU::F1:  return 1;
  case GPU::F2:  return 2;  case GPU::F3:  return 3;
  default: return 0;
  }
}

void GPUMCCodeEmitter::encode128(const MCInst &MI, uint32_t W[4]) const {
  unsigned Opcode = MI.getOpcode();
  const MCInstrDesc &Desc = MCII.get(Opcode);

  // Default all fields to zero
  uint8_t opcode = 0, pred = 0, flag_reg = 0, cond = 0;
  uint8_t sat = 0, imm_en = 0;
  uint8_t dst = 0, src0 = 0, src0_mod = 0;
  uint8_t src1 = 0, src1_mod = 0;
  uint8_t src2 = 0, src2_mod = 0;
  uint32_t imm32 = 0;

  // Extract opcode from the TableGen encoding
  // The opcode field is bits [127:120] of the 128-bit Inst
  uint64_t TSFlags = Desc.TSFlags;
  (void)TSFlags;

  // Map LLVM opcode to hardware opcode and extract operands
  switch (Opcode) {
  default:
    break;

  // NOP
  case GPU::NOP:
    opcode = 0x00;
    break;

  // MOV dst, src0
  case GPU::MOV:
    opcode = 0x01;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    break;

  // MOVI dst, imm32
  case GPU::MOVI:
    opcode = 0x02;
    imm_en = 1;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    imm32 = MI.getOperand(1).getImm();
    break;

  // ALU dst, src0, src1 (register forms)
  case GPU::ADD:  opcode = 0x03; goto alu_rrr;
  case GPU::SUB:  opcode = 0x04; goto alu_rrr;
  case GPU::MUL:  opcode = 0x05; goto alu_rrr;
  case GPU::AND:  opcode = 0x06; goto alu_rrr;
  case GPU::OR:   opcode = 0x07; goto alu_rrr;
  case GPU::XOR:  opcode = 0x08; goto alu_rrr;
  case GPU::SHL:  opcode = 0x0A; goto alu_rrr;
  case GPU::SHR:  opcode = 0x0B; goto alu_rrr;
  case GPU::SHRA: opcode = 0x0C; goto alu_rrr;
  case GPU::SMINrr: opcode = 0x10; goto alu_rrr;
  case GPU::SMAXrr: opcode = 0x11; goto alu_rrr;
  case GPU::UMINrr: opcode = 0x12; goto alu_rrr;
  case GPU::UMAXrr: opcode = 0x13; goto alu_rrr;
  case GPU::GETSR:
    opcode = 0x14;
    imm_en = 1;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    imm32 = MI.getOperand(1).getImm();
    break;
  case GPU::FADD: opcode = 0x40; goto float_rrr;
  case GPU::FMUL: opcode = 0x41; goto float_rrr;
  case GPU::FSUB: opcode = 0x42; goto float_rrr;
  case GPU::FDIV: opcode = 0x43; goto float_rrr;
  case GPU::FMIN: opcode = 0x44; goto float_rrr;
  case GPU::FMAX: opcode = 0x45; goto float_rrr;
  float_rrr:
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    src1 = getRegEncoding(MI.getOperand(2).getReg());
    // Source modifiers packed in MCInst flags by GPUMCInstLower
    src0_mod = MI.getFlags() & 0x3;
    src1_mod = (MI.getFlags() >> 2) & 0x3;
    break;
  alu_rrr:
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    src1 = getRegEncoding(MI.getOperand(2).getReg());
    break;

  // ALU dst, src0, imm32 (immediate forms)
  case GPU::ADDi:  opcode = 0x03; goto alu_rri;
  case GPU::SUBi:  opcode = 0x04; goto alu_rri;
  case GPU::MULi:  opcode = 0x05; goto alu_rri;
  case GPU::ANDi:  opcode = 0x06; goto alu_rri;
  case GPU::ORi:   opcode = 0x07; goto alu_rri;
  case GPU::XORi:  opcode = 0x08; goto alu_rri;
  case GPU::SHLi:  opcode = 0x0A; goto alu_rri;
  case GPU::SHRi:  opcode = 0x0B; goto alu_rri;
  case GPU::SHRAi: opcode = 0x0C; goto alu_rri;
  case GPU::FADDi: opcode = 0x40; goto alu_rri;
  case GPU::FMULi: opcode = 0x41; goto alu_rri;
  case GPU::FSUBi: opcode = 0x42; goto alu_rri;
  alu_rri:
    imm_en = 1;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    imm32 = MI.getOperand(2).getImm();
    break;

  // Unary dst, src0
  case GPU::NOT:  opcode = 0x09; goto unary;
  case GPU::NEG:  opcode = 0x0D; goto unary;
  case GPU::ITOF:  opcode = 0x46; goto unary;
  case GPU::FTOI:  opcode = 0x47; goto unary;
  case GPU::UITOF: opcode = 0x49; goto unary;
  case GPU::FTOU:  opcode = 0x4A; goto unary;
  unary:
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    break;

  // FMA dst, src0, src1, src2
  case GPU::FMA:
    opcode = 0x48;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    src1 = getRegEncoding(MI.getOperand(2).getReg());
    src2 = getRegEncoding(MI.getOperand(3).getReg());
    break;

  // REDUCE dst, src0, reduce_op
  case GPU::REDUCE:
    opcode = 0x0F;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    cond = MI.getOperand(2).getImm();
    break;

  // SEL dst, src0, src1, freg
  case GPU::SEL:
    opcode = 0x0E;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    src1 = getRegEncoding(MI.getOperand(2).getReg());
    flag_reg = MI.getOperand(3).getImm();
    break;

  // SELi dst, src0, imm32, freg
  case GPU::SELi:
    opcode = 0x0E;
    imm_en = 1;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    imm32 = MI.getOperand(2).getImm();
    flag_reg = MI.getOperand(3).getImm();
    break;

  // Memory: LDV dst, beat
  case GPU::LDV:
    opcode = 0x20;
    imm_en = 1;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    imm32 = MI.getOperand(1).getImm();
    break;

  // Memory: STV src0, beat
  case GPU::STV:
    opcode = 0x21;
    imm_en = 1;
    src0 = getRegEncoding(MI.getOperand(0).getReg());
    imm32 = MI.getOperand(1).getImm();
    break;

  // Memory: LD_SCALAR dst, src0, offset (broadcast to all lanes)
  case GPU::LD_SCALAR:
    opcode = 0x26;
    imm_en = 1;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    imm32 = MI.getOperand(2).getImm();
    break;

  // Memory: LD_SCATTER dst, src0, offset
  case GPU::LD_SCATTER:
    opcode = 0x22;
    imm_en = 1;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    imm32 = MI.getOperand(2).getImm();
    break;

  // Memory: LD_LOCAL dst, src0, offset
  case GPU::LD_LOCAL:
    opcode = 0x27;
    imm_en = 1;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    imm32 = MI.getOperand(2).getImm();
    break;

  // Memory: ST_SCATTER src0, src1, offset
  case GPU::ST_SCATTER:
    opcode = 0x23;
    imm_en = 1;
    src0 = getRegEncoding(MI.getOperand(0).getReg());
    src1 = getRegEncoding(MI.getOperand(1).getReg());
    imm32 = MI.getOperand(2).getImm();
    break;

  // Memory: ST_LOCAL src0, src1, offset
  case GPU::ST_LOCAL:
    opcode = 0x28;
    imm_en = 1;
    src0 = getRegEncoding(MI.getOperand(0).getReg());
    src1 = getRegEncoding(MI.getOperand(1).getReg());
    imm32 = MI.getOperand(2).getImm();
    break;

  // Memory: ATOMIC dst, src0, src1, offset, atomic_op
  case GPU::ATOMIC:
    opcode = 0x24;
    imm_en = 1;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    src1 = getRegEncoding(MI.getOperand(2).getReg());
    imm32 = MI.getOperand(3).getImm();
    cond = MI.getOperand(4).getImm();
    break;

  // Memory: ATOMIC_LOCAL dst, src0, src1, offset, atomic_op
  case GPU::ATOMIC_LOCAL:
    opcode = 0x29;
    imm_en = 1;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    src1 = getRegEncoding(MI.getOperand(2).getReg());
    imm32 = MI.getOperand(3).getImm();
    cond = MI.getOperand(4).getImm();
    break;

  // Memory: ATOMIC_CAS dst, src0(addr), src1(cmp), src2(swap), imm32(offset)
  case GPU::ATOMIC_CAS:
    opcode = 0x24;
    imm_en = 1;
    cond = 9;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    src1 = getRegEncoding(MI.getOperand(2).getReg());
    src2 = getRegEncoding(MI.getOperand(3).getReg());
    imm32 = MI.getOperand(4).getImm();
    break;

  // Memory: ATOMIC_LOCAL_CAS dst, src0(addr), src1(cmp), src2(swap), imm32(offset)
  case GPU::ATOMIC_LOCAL_CAS:
    opcode = 0x29;
    imm_en = 1;
    cond = 9;
    dst = getRegEncoding(MI.getOperand(0).getReg());
    src0 = getRegEncoding(MI.getOperand(1).getReg());
    src1 = getRegEncoding(MI.getOperand(2).getReg());
    src2 = getRegEncoding(MI.getOperand(3).getReg());
    imm32 = MI.getOperand(4).getImm();
    break;

  // Memory: PIXEL_OUT src0, src1, src2
  case GPU::PIXEL_OUT:
    opcode = 0x25;
    src0 = getRegEncoding(MI.getOperand(0).getReg());
    src1 = getRegEncoding(MI.getOperand(1).getReg());
    src2 = getRegEncoding(MI.getOperand(2).getReg());
    dst = 31; // ZERO_REG
    break;

  // Compare: CMP src0, src1, cc, freg
  case GPU::CMPrr:
    opcode = 0x60;
    src0 = getRegEncoding(MI.getOperand(0).getReg());
    src1 = getRegEncoding(MI.getOperand(1).getReg());
    cond = MI.getOperand(2).getImm();
    flag_reg = MI.getOperand(3).getImm();
    break;

  case GPU::CMPri:
    opcode = 0x60;
    imm_en = 1;
    src0 = getRegEncoding(MI.getOperand(0).getReg());
    imm32 = MI.getOperand(1).getImm();
    cond = MI.getOperand(2).getImm();
    flag_reg = MI.getOperand(3).getImm();
    break;

  // Control flow
  // GOTO: flag-conditional diverge. flag_reg from operand 0, pred from
  // operand 1 (GPUPredMode: 1=PRED_IF, 2=PRED_IF_NOT), imm32 from operand 2.
  case GPU::GOTO_INST:
    opcode = 0x80;
    imm_en = 1;
    flag_reg = getRegEncoding(MI.getOperand(0).getReg());
    pred = MI.getOperand(1).getImm();
    imm32 = MI.getOperand(2).getImm();
    break;

  // JOIN: reconverge (pop stack). No operands.
  case GPU::JOIN_INST:
    opcode = 0x81;
    break;

  // WHILE: loop init (push empty). No operands.
  case GPU::WHILE_INST:
    opcode = 0x82;
    break;

  // BREAK: accumulate exiting lanes into compiler-selected stack entry.
  // flag_reg from operand 0, pred from operand 1 (GPUPredMode), cond from
  // operand 2 (relative target depth), imm32 from operand 3.
  case GPU::BREAK_INST:
    opcode = 0x83;
    imm_en = 1;
    flag_reg = getRegEncoding(MI.getOperand(0).getReg());
    pred = MI.getOperand(1).getImm();
    cond = MI.getOperand(2).getImm();
    imm32 = MI.getOperand(3).getImm();
    break;

  // JUMP: unconditional branch. imm32 from operand 0.
  case GPU::JUMP_INST:
    opcode = 0x84;
    imm_en = 1;
    imm32 = MI.getOperand(0).getImm();
    break;

  case GPU::HALT:
  case GPU::HALT_RET:
    opcode = 0x85;
    break;

  case GPU::BARRIER:
    opcode = 0x86;
    cond = MI.getOperand(0).getImm();
    break;

  case GPU::MEM_FENCE:
    opcode = 0x87;
    cond = MI.getOperand(0).getImm();
    break;
  }

  // Encode matching make_instr128() layout exactly:
  // w[3]: opcode[31:24] | pred[23:22] | flag_reg[21:20] | cond[19:16] |
  //       sat[15] | imm_en[14] | rsvd[13:12] | dst[11:6] | src0[5:0]
  W[3] = ((uint32_t)(opcode & 0xFF) << 24) |
         ((uint32_t)(pred & 0x3) << 22) |
         ((uint32_t)(flag_reg & 0x3) << 20) |
         ((uint32_t)(cond & 0xF) << 16) |
         ((uint32_t)(sat & 0x1) << 15) |
         ((uint32_t)(imm_en & 0x1) << 14) |
         ((uint32_t)(dst & 0x3F) << 6) |
         ((uint32_t)(src0 & 0x3F));

  // w[2]: src0_mod[31:30] | src1[29:24] | src1_mod[23:22] |
  //       src2[21:16] | src2_mod[15:14] | rsvd[13:0]
  W[2] = ((uint32_t)(src0_mod & 0x3) << 30) |
         ((uint32_t)(src1 & 0x3F) << 24) |
         ((uint32_t)(src1_mod & 0x3) << 22) |
         ((uint32_t)(src2 & 0x3F) << 16) |
         ((uint32_t)(src2_mod & 0x3) << 14);

  // w[1]: imm32
  W[1] = imm32;

  // w[0]: reserved (zero)
  W[0] = 0;
}

void GPUMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                         SmallVectorImpl<char> &CB,
                                         SmallVectorImpl<MCFixup> &Fixups,
                                         const MCSubtargetInfo &STI) const {
  // Skip pseudo instructions
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  if (Desc.isPseudo())
    return;

  uint32_t W[4];
  encode128(MI, W);

  // Emit as 4 little-endian uint32_t words (matching DDR byte order)
  for (int i = 0; i < 4; i++) {
    uint32_t Val = W[i];
    CB.push_back(Val & 0xFF);
    CB.push_back((Val >> 8) & 0xFF);
    CB.push_back((Val >> 16) & 0xFF);
    CB.push_back((Val >> 24) & 0xFF);
  }
}

MCCodeEmitter *llvm::createGPUMCCodeEmitter(const MCInstrInfo &MCII,
                                            MCContext &Ctx) {
  return new GPUMCCodeEmitter(MCII, Ctx);
}
