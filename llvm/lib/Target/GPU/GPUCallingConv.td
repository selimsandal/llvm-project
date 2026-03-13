//===-- GPUCallingConv.td - GPU Calling Convention --*- tablegen -*-===//
//
// The GPU has no traditional calling convention (no call/ret, everything
// is inlined). This is a minimal definition required by LLVM infrastructure.
//
//===--------------------------------------------------------------===//

// All values returned in R5 (first allocatable register)
def RetCC_GPU : CallingConv<[
  CCIfType<[i32, f32], CCAssignToReg<[R5]>>
]>;

// Arguments passed in R1-R4 (descriptor init registers)
def CC_GPU : CallingConv<[
  CCIfType<[i32, f32], CCAssignToReg<[R1, R2, R3, R4]>>
]>;
