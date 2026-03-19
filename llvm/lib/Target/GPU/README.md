# GPU Backend Notes

This backend targets the custom SIMT GPU in the superproject. The ISA is
explicitly structured and compiler-controlled: `GOTO`, `JOIN`, `WHILE`,
`BREAK(depth,target)`, and `JUMP` are real instructions, and the hardware keeps
only an `exec_mask` plus a plain mask stack.

This README owns compiler/backend-specific decisions. The superproject root
`README` should only carry project-wide state.

## Current Compute ABI Direction

The current compiler-facing workgroup ABI is:

- system values come from hidden workgroup context through `I_GETSR`
- raw state exposed to the compiler is:
  - `group_id_x/y/z`
  - `local_size_x/y/z`
  - `num_groups_x/y/z`
  - `local_id_x/y/z`
  - subgroup-local base coordinates when needed internally
- `global_id` is compiler-derived:
  - `group_id * local_size + local_id`

The backend should not grow new convenience `global_id` special-register
surfaces unless there is a proven need.

## Current Compute Frontend Status

Working on the current compute subset:

- OpenCL builtins are moving onto raw workgroup state + compiler-derived IDs
- HLSL system values are moving onto the same path
- direct LLVM IR `addrspace(3)` loads/stores/atomics lower to
  `LD_LOCAL` / `ST_LOCAL` / `ATOMIC_LOCAL`
- compiler-visible barrier/fence lowering is real

Currently supported source-level pieces on the intended path:

- OpenCL `barrier()` / `mem_fence()` for the supported subset
- optimized OpenCL `__local` kernel arguments
- optimized OpenCL local atomic builtins on the currently verified subset:
  `atomic_add`, `atomic_xchg`, `atomic_or`, `atomic_and`, `atomic_xor`,
  `atomic_min`, `atomic_max`, `atomic_cmpxchg`
- HLSL `groupshared` globals plus
  `GroupMemoryBarrierWithGroupSync()` for the current compute subset
- HLSL `groupshared` `InterlockedAdd/And/Or/Xor/Min/Max/Exchange/
  CompareExchange` for the current compute subset

Still incomplete:

- full simulator-sync parity for these higher-level surfaces
- any source-level atomic variants outside the currently verified subset

## Current Review Notes

These notes explain the current LLVM-side edits and why they belong in the
submodule instead of the superproject.

### Why LLVM Submodule Edits Were Necessary

These changes live in `External/llvm-project` because the behavior being
changed is owned there:

- Clang headers decide whether source-level HLSL intrinsics are even declared
- LLVM intrinsics define the IR contract the GPU backend lowers
- the GPU backend owns instruction selection, address-space handling, builtin
  lowering, and machine-code emission

The superproject cannot fake these semantics from scripts or host runtime code.
If the compiler is supposed to emit different IR or machine instructions, the
authoritative place to change that is the LLVM submodule.

### File-By-File Rationale

- `clang/lib/Headers/hlsl/hlsl_alias_intrinsics.h`
  - added `Interlocked*` declarations for `groupshared` scalars
  - reason: source-level HLSL cannot exercise groupshared atomics if Clang does
    not declare those builtins in the default header path

- `llvm/include/llvm/IR/IntrinsicsDirectX.td`
  - marked `dx.group.memory.barrier.with.group.sync` with memory effects
  - reason: the IR needs to describe that this operation orders memory and is
    convergent, otherwise optimization/lowering can treat it too weakly

- `llvm/include/llvm/IR/IntrinsicsGPU.td`
  - added `llvm.gpu.getsr`, `llvm.gpu.workgroup.sync`, and
    `llvm.gpu.mem.fence`
  - reason: the new GPU ABI should not be modeled with ad hoc
    `llvm.read_register("rN")` strings or frontend-specific shims

- `llvm/lib/Target/GPU/GPUTargetMachine.cpp`
  - added the local-global allocation pass for `addrspace(3)` globals
  - stopped flattening local/shared memory into flat global memory
  - reason: the backend cannot emit honest `LD_LOCAL` / `ST_LOCAL` /
    `ATOMIC_LOCAL` if local/shared memory is erased before instruction
    selection

- `llvm/lib/Target/GPU/GPUInstrInfo.td`
  - added machine instruction definitions for `GETSR`, `LD_LOCAL`, `ST_LOCAL`,
    `ATOMIC_LOCAL`, `ATOMIC_LOCAL_CAS`, `BARRIER`, `MEM_FENCE`, and `HALT_RET`
  - reason: these are the compiler-visible instructions required by the new
    compute ABI and simulator/runtime bring-up
  - reviewer note: `HALT_RET` is not a new hardware opcode; it keeps a return
    register operand alive through lowering while encoding the same `halt`

- `llvm/lib/Target/GPU/MCTargetDesc/GPUMCCodeEmitter.cpp`
  - added actual binary encoding for those instructions
  - reason: instruction defs alone are not enough; the backend still has to
    emit the right 128-bit ISA words

- `llvm/lib/Target/GPU/GPUISelLowering.h`
  - added the `GETSR` target DAG node declaration
  - reason: the backend needs an explicit internal node to carry special-register
    reads from IR intrinsic lowering into instruction selection

- `llvm/lib/Target/GPU/GPUISelLowering.cpp`
  - lowers `llvm.gpu.getsr` to `GPUISD::GETSR`
  - adds return-lowering support so return values stay live to `HALT_RET`
  - reason: special-register reads and the updated test coverage need proper
    lowering through the target DAG

- `llvm/lib/Target/GPU/GPUISelDAGToDAG.cpp`
  - selects `addrspace(3)` loads/stores to `LD_LOCAL` / `ST_LOCAL`
  - selects local-vs-global atomics correctly
  - selects `llvm.gpu.getsr`, `llvm.gpu.workgroup.sync`, and
    `llvm.gpu.mem.fence`
  - handles `HALT_RET`
  - reason: this is the selector layer that decides which GPU ISA instruction
    a given IR/DAG operation becomes

- `llvm/lib/Target/GPU/GPUSPIRVLowering.cpp`
  - moved OpenCL builtin IDs onto raw workgroup state + compiler-derived
    `global_id`
  - added lowering for sync builtins and broader atomic builtins
  - added alloca promotion and wrapper-body cloning to deal with the OpenCL
    wrapper/generated-function shape
  - reviewer note: this file now explicitly ignores non-declaration functions
    during builtin classification so real functions are not misidentified and
    deleted just because their names contain builtin substrings

- `llvm/lib/Target/GPU/GPUHLSLLowering.cpp`
  - moved HLSL system values to raw workgroup state + compiler-derived IDs
  - added lowering for `GroupMemoryBarrierWithGroupSync()`
  - added lowering for `groupshared` `Interlocked*`
  - added simple-allocation promotion after lowering
  - reason: HLSL has its own frontend intrinsic shape, so these changes belong
    in the HLSL lowering pass instead of the OpenCL/SPIR-V path

- `llvm/lib/Target/GPU/README.md`
  - updated to reflect the current backend ABI and source-level status
  - reason: reviewers need the local compiler contract documented next to the
    backend, not inferred from the superproject root README

### Test Additions And Why They Exist

These tests were added because each new lowering path needs direct backend
coverage, not just higher-level host tests.

- `llvm/test/CodeGen/GPU/hlsl-thread-ids.ll`
  - proves HLSL system-value lowering uses raw workgroup state / derived IDs
- `llvm/test/CodeGen/GPU/hlsl-groupshared-sync.ll`
  - proves `GroupMemoryBarrierWithGroupSync()` lowers to GPU sync ops
- `llvm/test/CodeGen/GPU/hlsl-local-atomics.ll`
  - proves `groupshared` `Interlocked*` lowering reaches atomics correctly
- `llvm/test/CodeGen/GPU/local-memory.ll`
  - proves direct IR local load/store lowering
- `llvm/test/CodeGen/GPU/local-atomic.ll`
  - proves direct IR local atomic lowering
- `llvm/test/CodeGen/GPU/sync.ll`
  - proves direct IR sync intrinsic lowering
- `llvm/test/CodeGen/GPU/opencl-workgroup-builtins.ll`
  - proves OpenCL builtin IDs use the new workgroup ABI
- `llvm/test/CodeGen/GPU/opencl-sync-builtins.ll`
  - proves OpenCL sync builtin lowering
- `llvm/test/CodeGen/GPU/opencl-local-arg.ll`
  - proves optimized OpenCL `__local` kernel arguments reach local-memory ops
- `llvm/test/CodeGen/GPU/opencl-local-arg-o0.ll`
  - exists specifically to keep the unoptimized `-O0` local-argument wrapper
    path from regressing again
- `llvm/test/CodeGen/GPU/opencl-local-atomic-builtins.ll`
  - proves source-level OpenCL local atomic builtins lower to local atomics

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
2. Superproject host-side object verification (`Source/Host/Tests/compiler_verify.c`)
3. Superproject ISA simulator (`Source/Host/Lib/gpu_sim.c`)
4. Superproject pathtracer compare harness
   (`Source/Host/PathTracer/pathtracer_compare.c`)

Interpretation:

- `sim != x86 HLSL`: compiler/object semantics bug
- `sim == x86 HLSL` but `fpga != sim`: RTL, stale bitstream, or board-state bug

The backend is intentionally moving away from late clever CFG surgery. If a new
change requires complex post-RA reconstruction to make control flow work, the
preferred fix is usually to make the IR/MI entering `GPUControlFlow.cpp` more
structured instead.
