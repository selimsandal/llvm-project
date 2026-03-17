# GPU Backend Notes

This backend targets the custom SIMT GPU in the superproject. The ISA is
explicitly structured and compiler-controlled: `GOTO`, `JOIN`, `WHILE`,
`BREAK(depth,target)`, and `JUMP` are real instructions, and the hardware keeps
only an `exec_mask` plus a plain mask stack.

## Pipeline

1. IR is cleaned up before ISel in `GPUTargetMachine.cpp` with
   `FixIrreducible`, `UnifyLoopExits`, and `StructurizeCFG`.
2. Normal instruction selection lowers scalar LLVM IR to GPU machine
   instructions.
3. `GPUControlFlow.cpp` converts structured machine CFG regions into
   `WHILE/BREAK/JUMP/JOIN` and `GOTO/JOIN`.
4. `GPUPeephole.cpp` performs late local combines and patches final branch
   offsets plus compiler-selected BREAK depths from the final nesting stack.

The important rule is that hardware does not search for loop frames. The
compiler decides which mask-stack frame a BREAK targets and encodes that depth
directly in the instruction.

## Control Flow Contract

- `GOTO` always pushes a mask-stack entry, even if the active mask is zero.
- `JOIN` pops one entry and ORs it back into `exec_mask`.
- `WHILE` pushes an empty frame.
- `BREAK` accumulates its active lanes into the compiler-selected stack depth,
  clears them from `exec_mask`, and jumps to the nearest enclosing `JOIN` when
  the current `exec_mask` drains to zero.
- `JUMP` is the unconditional back-edge.

This keeps the hardware simple and pushes all loop-targeting policy into the
compiler.

## Debugging

Use the validation stack in this order:

1. `./Scripts/test-compiler.sh`
2. Superproject host-side object verification (`Source/Host/tests/compiler_verify.c`)
3. Superproject ISA simulator (`Source/Host/gpu_sim.c`)
4. Superproject pathtracer compare harness
   (`Source/Host/pathtracer/pathtracer_compare.c`)

Interpretation:

- `sim != x86 HLSL`: compiler/object semantics bug
- `sim == x86 HLSL` but `fpga != sim`: RTL, stale bitstream, or board-state bug

The backend is intentionally moving away from late clever CFG surgery. If a new
change requires complex post-RA reconstruction to make control flow work, the
preferred fix is usually to make the IR/MI entering `GPUControlFlow.cpp` more
structured instead.
