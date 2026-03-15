//===-- GPURegisterInfo.td - GPU Register defs ---*- tablegen -*-===//
//
// Register file definitions for the custom GPU.
// 32 x 32-bit GPRs (r0-r31), 4 x 8-bit flag registers (f0-f3).
//
//===--------------------------------------------------------------===//

// Base register class
class GPUReg<bits<16> enc, string n> : Register<n> {
  let HWEncoding = enc;
  let Namespace = "GPU";
}

//===--------------------------------------------------------------===//
// General Purpose Registers (32-bit)
//===--------------------------------------------------------------===//

def R0  : GPUReg<0,  "r0">;
def R1  : GPUReg<1,  "r1">;
def R2  : GPUReg<2,  "r2">;
def R3  : GPUReg<3,  "r3">;
def R4  : GPUReg<4,  "r4">;
def R5  : GPUReg<5,  "r5">;
def R6  : GPUReg<6,  "r6">;
def R7  : GPUReg<7,  "r7">;
def R8  : GPUReg<8,  "r8">;
def R9  : GPUReg<9,  "r9">;
def R10 : GPUReg<10, "r10">;
def R11 : GPUReg<11, "r11">;
def R12 : GPUReg<12, "r12">;
def R13 : GPUReg<13, "r13">;
def R14 : GPUReg<14, "r14">;
def R15 : GPUReg<15, "r15">;
def R16 : GPUReg<16, "r16">;
def R17 : GPUReg<17, "r17">;
def R18 : GPUReg<18, "r18">;
def R19 : GPUReg<19, "r19">;
def R20 : GPUReg<20, "r20">;
def R21 : GPUReg<21, "r21">;
def R22 : GPUReg<22, "r22">;
def R23 : GPUReg<23, "r23">;
def R24 : GPUReg<24, "r24">;
def R25 : GPUReg<25, "r25">;
def R26 : GPUReg<26, "r26">;
def R27 : GPUReg<27, "r27">;
def R28 : GPUReg<28, "r28">;
def R29 : GPUReg<29, "r29">;
def R30 : GPUReg<30, "r30">;
def R31 : GPUReg<31, "r31">;

// GPR register class: allocatable regs first, special regs last
// r0 = thread ID (reserved, read-only)
// r1-r4 = descriptor init (allocatable after prologue)
// r5-r29, r31 = general purpose
// r30 = stack pointer (reserved)
def GPR : RegisterClass<"GPU", [i32, f32], 32,
            (add R5, R6, R7, R8, R9, R10, R11, R12,
             R13, R14, R15, R16, R17, R18, R19, R20,
             R21, R22, R23, R24, R25, R26, R27, R28,
             R29, R31,
             R1, R2, R3, R4,
             R0, R30)>;

//===--------------------------------------------------------------===//
// Flag Registers (8-bit, per-lane predicate masks)
//===--------------------------------------------------------------===//

def F0 : GPUReg<0, "f0">;
def F1 : GPUReg<1, "f1">;
def F2 : GPUReg<2, "f2">;
def F3 : GPUReg<3, "f3">;

def FlagRC : RegisterClass<"GPU", [i32], 32, (add F0, F1, F2, F3)>;
