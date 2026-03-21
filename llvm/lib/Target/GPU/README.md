# GPU Backend Notes

This backend targets the custom SIMT GPU in the superproject. The ISA is
explicitly structured and compiler-controlled: `GOTO`, `JOIN`, `WHILE`,
`BREAK(depth,target)`, and `JUMP` are real instructions, and the hardware keeps
only an `exec_mask` plus a plain mask stack.

This README owns compiler/backend-specific decisions. The superproject root
[CLAUDE.md](/home/selimsandal/Developer/gpu/CLAUDE.md) should carry internal
project-wide state.

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
- compiler-emitted `.gpu.meta` launch metadata for:
  - fixed local size when declared by the frontend
  - hidden workgroup-context requirement
  - basic local-memory / atomic / barrier / fence feature bits
  - original kernel arg count plus direct local-arg mask for the `<=4`-arg path

Still incomplete:

- full simulator-sync parity for these higher-level surfaces
- any source-level atomic variants outside the currently verified subset
- richer reflection beyond the current minimal launch record

## Compilation Pipelines

Working:

- `LLVM IR -> llc -march=gpu -> ELF .o -> gpu_shader_loader.h`
  - write kernels in LLVM IR with `r1-r4` as arguments
  - `<=4` args use direct registers
  - `>4` args use an indirect buffer in `r1`
  - `r0` remains the physical thread ID for low-level code, but
    compiler-visible workgroup/system values lower through hidden workgroup
    context plus `I_GETSR`
- `OpenCL C -> clang -target spir -> llvm-spirv -> llc -march=gpu`
  - current compute subset only
  - `mad()` / `min()` / `max()` lower cleanly
  - builtin IDs lower through hidden workgroup context / `I_GETSR` with
    compiler-derived `global_id`
  - source-level `barrier()` / `mem_fence()` lower to real GPU sync ops
  - optimized `__local` kernel arguments lower onto the local-memory path
  - current runtime ABI for `__local` pointer args is a byte offset into
    per-workgroup local memory, not a DDR pointer
- `HLSL -> clang -x hlsl -> LLVM IR -> GPUHLSLLowering -> llc -march=gpu`
  - current compute-only subset
  - `@llvm.dx.*` system values lower through hidden workgroup context /
    `I_GETSR` and compiler-derived IDs
  - resource bindings still map to `r1-r4` (`<=4` direct, `>4` indirect)
  - `groupshared` globals, `GroupMemoryBarrierWithGroupSync()`, and the
    verified `groupshared` `Interlocked*` subset compile on this path

Blocked:

- DXC and Slang both produce Vulkan SPIR-V, which the current
  SPIRV-LLVM-Translator path cannot ingest
- Slang's `llvm-shader-ir` target emits host-style x86 IR with explicit thread
  loops, not GPU shader IR

## External Tools

| Tool | Location | Use |
|------|----------|-----|
| DXC | In `PATH` (`dxc`) | HLSL -> SPIR-V (Vulkan, currently blocked by the translator) |
| Slang | In `PATH` (`slangc`) | Slang -> SPIR-V (same Vulkan issue) or host-style LLVM IR |
| llvm-spirv | not yet built | SPIR-V <-> LLVM IR (OpenCL SPIR-V only) |
| llc | `External/llvm-project/build/bin/llc` | LLVM IR -> GPU asm / obj |
| gpu-compiler | `External/llvm-project/build/bin/gpu-compiler` | standalone LLVM IR -> GPU ELF compiler |

## Key Backend Files

| File | Purpose |
|------|---------|
| `GPUISelLowering.cpp` | DAG lowering: `BR_CC` -> `CMP+BRCOND`, `SELECT_CC` -> `CMP+SEL`, f32/i32 ops |
| `GPUISelDAGToDAG.cpp` | DAG -> MI selection: `CMP`, `SEL`, `BRCOND`, `BR`, `MOVI`, `RETURN` |
| `GPUInstrInfo.td` | instruction definitions, patterns, and pseudos such as `GPU_BRCOND`, `GPU_BR`, `LOOP_EXIT_MOVI`, `SELi` |
| `GPUInstrFormats.td` | 128-bit instruction encoding format matching `make_instr128()` |
| `GPUTargetMachine.cpp` | pre-ISel pipeline setup, target pass pipeline, and metadata emission placement |
| `GPUControlFlow.cpp` | post-RA structured control-flow lowering |
| `GPUPeephole.cpp` | post-RA local combines and final branch/BREAK patching |
| `GPUMCCodeEmitter.cpp` | 128-bit binary encoding |
| `GPUMCInstLower.cpp` | MI -> MCInst with source modifier flags |
| `GPUSPIRVLowering.cpp` | OpenCL/SPIR-V builtin lowering onto raw workgroup state + compiler-derived IDs |
| `GPUHLSLLowering.cpp` | HLSL lowering: system values, resources, wave ops, barriers, and `groupshared` intrinsics |
| `GPUKernelMetadata.cpp` | `.gpu.meta` section emission |

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
  - now also runs the kernel-metadata emission pass after frontend lowering and
    local-global allocation
  - reason: the backend cannot emit honest `LD_LOCAL` / `ST_LOCAL` /
    `ATOMIC_LOCAL` if local/shared memory is erased before instruction
    selection, and the host cannot stop hand-filling launch metadata unless the
    object carries a stable reflection record

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
  - intentionally keeps `hlsl.shader` / `hlsl.numthreads` function attributes
    alive long enough for the metadata-emission pass to reflect them into
    `.gpu.meta`
  - reason: HLSL has its own frontend intrinsic shape, so these changes belong
    in the HLSL lowering pass instead of the OpenCL/SPIR-V path

- `llvm/lib/Target/GPU/GPUKernelMetadata.cpp`
  - new module pass that emits the `.gpu.meta` ELF section
  - reason: launch metadata belongs in the compiled object, not in duplicated
    host-side tables or sidecar files
  - design decision:
    - phase 1 records only launch-relevant facts shared by both OpenCL and HLSL
    - fixed local size is only reflected when the frontend declares one
      (`hlsl.numthreads` or OpenCL `reqd_work_group_size`)
    - dynamic OpenCL local size remains a runtime choice

- `llvm/lib/Target/GPU/GPUControlFlow.cpp`
  - keeps the reverse/forward leaf-first conditional scan and the
    triangle-before-diamond preference
  - preserves an outer condition only when the same flag is clobbered in the
    trailing straight-line tail of the true path
  - explicitly does not preserve through already-lowered nested structured
    control flow in the same block
  - reason: the mandelbrot loop needs the straight-line clobber case, while
    PathTracer regressed badly when nested compares inside already-structured
    regions were treated as if they still belonged to the outer branch

- `llvm/lib/Target/GPU/GPURegisterInfo.cpp`
  - keeps only `r0` and `r30` globally reserved
  - reason: globally reserving an extra late control-flow scratch register
    perturbed PathTracer register allocation and changed code shape enough to
    create false semantic mismatches in the compare harness

- `llvm/lib/Target/GPU/README.md`
  - updated to reflect the current backend ABI and source-level status
  - reason: reviewers need the local compiler contract documented next to the
    backend, not inferred from the superproject root `CLAUDE.md`

### Test Additions And Why They Exist

These tests were added because each new lowering path needs direct backend
coverage, not just higher-level host tests.

The new `.gpu.meta` object section is currently validated end to end through
the superproject host loader/simulator tests rather than a dedicated LLVM lit
object-section test. That is intentional for now: the guaranteed local tool
surface in this repo is `llc`, `gpu-compiler`, `clang`, and `FileCheck`, while
the host harness already proves the full chain of section emission, ELF
loading, reflected descriptor build, and execution.

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
- `llvm/test/CodeGen/GPU/signed-int-compare.ll`
  - now matches the current signed-compare lowering shape
  - reason: the backend intentionally expands signed compares through
    `SMIN/SMAX` plus compares/selects rather than pretending there is a native
    signed-branch instruction
- `llvm/test/CodeGen/GPU/control-flow-comprehensive.ll`
  - now matches the current triangle lowering shape
  - reason: the backend currently prefers a branchless select form for that
    simple triangle instead of forcing a `goto/join` pair

## Validation Surface

Current GPU lit suite (`30` tests):

- `alu.ll`
- `atomic.ll`
- `branch-direction.ll`
- `control-flow-comprehensive.ll`
- `control-flow.ll`
- `float.ll`
- `hlsl-groupshared-sync.ll`
- `hlsl-local-atomics.ll`
- `hlsl-math.ll`
- `hlsl-thread-ids.ll`
- `hlsl-vec-add.ll`
- `hlsl-wave-reduce.ll`
- `local-atomic.ll`
- `local-memory.ll`
- `loop-break-divergent.ll`
- `loop.ll`
- `memory.ll`
- `movi-fold.ll`
- `opencl-local-arg-o0.ll`
- `opencl-local-arg.ll`
- `opencl-local-atomic-builtins.ll`
- `opencl-sync-builtins.ll`
- `opencl-workgroup-builtins.ll`
- `reduce.ll`
- `select.ll`
- `signed-int-compare.ll`
- `source-modifiers.ll`
- `spill.ll`
- `sync.ll`
- `uitof-ftou.ll`

Useful superproject host-side checks alongside the LLVM lit suite:

- `Source/Host/Tests/compiler_verify.c`
- `Source/Host/Tests/gpu_sim_test.c`
- `Source/Host/Tests/break_color_test.c`
- `Source/Host/Tests/rejection_loop_test.c`
- `Source/Host/Tests/gpu_hw_test.c`
- `Source/Host/PathTracer/pathtracer_compare.c`

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

## Peephole Optimizations

- FMA formation:
  `FMUL(t,a,b) + FADD(d,t,c) -> FMA(d,a,b,c)` when `t` is single-use and
  multiply operands are not clobbered before the add
- immediate folding:
  `MOVI(t,imm) + ALU(d,x,t) -> ALUi(d,x,imm)` for the supported ALU ops, with
  a multiple-definition guard to avoid folding across divergent control-flow
  shapes
- source modifier folding:
  `FSUB(MOVI(0),x) -> NEG` and `ANDi(x,0x7FFFFFFF) -> ABS`

## Current Limitations

Compiler/runtime gaps that still matter:

- source-level atomic variants outside the currently verified subset
- richer reflection/runtime metadata beyond the current minimal `.gpu.meta`
  launch record
- full simulator memory-model fidelity beyond the current one-workgroup
  barrier-rendezvous subset

Target/backend gaps that still need implementation:

- descriptor-relative `LDV` / `STV` ISel patterns for vertex/fragment flows
- `PIXEL_OUT` lowering from LLVM IR
- full vertex/fragment shader calling conventions and resource mapping

External-tool limitations:

- DXC and Slang produce Vulkan SPIR-V, which the current translator path
  rejects (`OpTypeForwardPointer`, `GLSL.std.450`,
  `SPV_KHR_storage_buffer_storage_class`)
- Slang `llvm-shader-ir` is host-style looped IR, not GPU shader IR

Hardware limitations that still surface at the backend boundary:

- no integer division (`SDIV` / `UDIV` expand to libcalls)
- no `FSQRT`, `FSIN`, `FCOS`, or `FPOW`
- `HALT` terminates all lanes, not individual lanes

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
