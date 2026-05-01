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

## HLSL Math Lowering Status

This section tracks how HLSL math reaches the GPU ISA. The important split is
whether the HLSL lowering pass emits an already-native LLVM operation, or
whether it synthesizes a sequence because the GPU has no dedicated opcode for
that HLSL intrinsic.

Native final ISA coverage already exists for these scalar operations:

| IR operation shape | Final GPU instruction | Notes |
|--------------------|-----------------------|-------|
| `llvm.fma.f32` | `FMA` | HLSL lowering emits this directly for `lerp` and dot-product accumulation; the post-RA peephole can also merge eligible `FMUL` + `FADD` pairs into `FMA` independently. |
| `llvm.sqrt.f32` | `FSQRT` | Used by `rsqrt` lowering before the reciprocal divide. |
| `llvm.maxnum.f32` / `llvm.minnum.f32` | `FMAX` / `FMIN` | Used by `nclamp` and `saturate`. |
| `llvm.smax/smin/umax/umin.i32` | `SMAX` / `SMIN` / `UMAX` / `UMIN` | Used by signed and unsigned clamp. |
| ordinary `fadd/fsub/fmul/fdiv` | `FADD` / `FSUB` / `FMUL` / `FDIV` | Used by expanded math formulas. |
| ordinary `add/sub/mul/and/or/xor/shl/lshr/ashr` | integer ALU ops | Used by integer math and scalar helper sequences. |
| `llvm.gpu.reduce_*` | `REDUCE` | Used by HLSL wave reductions and constant-lane `WaveReadLaneAt`. |

HLSL-specific math currently lowered in `GPUHLSLLowering.cpp`:

| HLSL / DX intrinsic | Lowering strategy | Native dependency |
|---------------------|-------------------|-------------------|
| `nclamp(x, lo, hi)` | `minnum(maxnum(x, lo), hi)` | `FMAX` + `FMIN` |
| `sclamp(x, lo, hi)` | `smin(smax(x, lo), hi)` | `SMAX` + `SMIN` |
| `uclamp(x, lo, hi)` | `umin(umax(x, lo), hi)` | `UMAX` + `UMIN` |
| `dot2/dot3/dot4` | first lane multiply, then an `llvm.fma` chain | `FMUL` + `FMA` |
| `fdot` for `float2/3/4` | extract vector lanes, then the same dot sequence | `FMUL` + `FMA` |
| `lerp(a, b, t)` | `llvm.fma(t, b - a, a)` | `FSUB` + `FMA` |
| `saturate(x)` | `minnum(maxnum(x, 0), 1)` | `FMAX` + `FMIN` |
| `frac(x)` | `x - floor(x)`; floor is built from truncation plus negative correction | `FTOI` + `ITOF` + compare/select + `FSUB` |
| `rsqrt(x)` | `1.0 / sqrt(x)` | `FSQRT` + `FDIV` |
| `imad/umad(a, b, c)` | `a * b + c` | integer `MUL` + `ADD` |
| `degrees(x)` | `x * (180 / pi)` | `FMUL` |
| `radians(x)` | `x * (pi / 180)` | `FMUL` |
| `sign(x)` | compare/select to `-1`, `0`, or `1` | compare + `SEL` |
| `step(y, x)` | `(x >= y) ? 1.0 : 0.0` | compare + `SEL` |

Scalar LLVM intrinsics that Clang can emit on the HLSL path are also rewritten
before instruction selection:

| LLVM intrinsic | Lowering strategy | Native dependency |
|----------------|-------------------|-------------------|
| `llvm.floor.f32` | truncation plus negative correction | `FTOI` + `ITOF` + compare/select |
| `llvm.ceil.f32` | truncation plus positive correction | `FTOI` + `ITOF` + compare/select |
| `llvm.trunc.f32` | float-to-int then int-to-float | `FTOI` + `ITOF` |
| `llvm.round.f32` | truncation plus half-away adjustment | `FTOI` + `ITOF` + compare/select |
| `llvm.roundeven.f32` | truncation plus ties-to-even adjustment | `FTOI` + `ITOF` + compare/select + integer parity test |
| `llvm.abs.i32` | compare/select between `x` and `-x` | integer ALU + `SEL` |

Current math gaps:

- the HLSL lowering does not implement transcendental/library-style intrinsics
  such as `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `exp`, `log`, or `pow`
- no native `RSQRT`, `FRAC`, `FLOOR`, `CEIL`, or `ROUND` opcode exists; those
  are synthesized as instruction sequences
- the explicit math lowering is scalar `i32` / `f32` focused; wider vectors,
  half, double, and i64 variants are only supported if earlier LLVM passes
  scalarize or rewrite them into the covered shapes
- this is the HLSL compute path only; OpenCL/SPIR-V math builtins are handled
  separately in `GPUSPIRVLowering.cpp`

## RTL-Supported / Compiler-Missing Feature Inventory

This section tracks hardware or host-runtime capabilities that exist below the
compiler but are not yet exposed as a general compiler feature. It is meant to
help choose future compiler work, not to imply that every RTL feature should be
lowered directly from LLVM.

| Area | RTL / host capability | Current compiler status | Compiler-side work if prioritized |
|------|-----------------------|-------------------------|-----------------------------------|
| ROP pixel output | RTL has `I_PIXEL_OUT` and `gpu_rop.sv` drains per-engine FIFOs, depth-tests, and writes color/depth to DDR. The 3D raster host path emits raster kernels that use this path. | `PIXEL_OUT` is defined in `GPUInstrInfo.td` and LLVM IR can emit it through the low-level `llvm.gpu.pixel.out(offset, depth, color)` intrinsic. There is not yet a source-level HLSL/OpenCL builtin. | If this becomes a source-facing feature, add a narrow builtin or HLSL lowering hook for software raster kernels and keep simulator/FPGA behavior aligned. |
| Triangle setup / raster pipeline | The superproject has a working software graphics pipeline: vertex transform, triangle setup, raster walk, and `PIXEL_OUT` into the ROP. | The compiler does not own a graphics pipeline. HLSL support is compute-only; there is no vertex/pixel shader stage lowering, no draw-call ABI, and no automatic triangle setup/raster generation. | Treat this as staged work: first expose `PIXEL_OUT` for compute-style raster kernels, then decide whether full graphics shader stages are worth modeling. |
| Buffer/resource binding model | Descriptors can initialize `r1-r4`; reflected launches can use an indirect argument buffer through `r1`; host code already builds descriptors from `.gpu.meta`. | HLSL/DX resources currently map binding slot `0..3` to `r1..r4`, and `>4` to the indirect args buffer. `RawBuffer`, simple `TypedBuffer`, and scalar `cbuffer` loads are covered, but binding space/range, descriptor arrays, and dynamic resource indexing are not real features. | Define the intended descriptor model first: whether binding `space`, ranges, arrays, and dynamic indexing become metadata, an explicit descriptor table in memory, or remain unsupported. Then teach `GPUHLSLLowering` / `GPUSPIRVLowering` to preserve and lower that model. |
| Constant buffers / cbuffers | Host paths can upload parameter blocks and pass the base address through a descriptor register. PathTracer now uses a real HLSL `cbuffer` for scalar params. | Simple scalar cbuffer member loads and `dx.resource.load.cbufferrow` lower to memory loads from the bound base. Full HLSL cbuffer layout support is not modeled: vectors/matrices, nested aggregates, arrays, packing edge cases, and richer reflection are still limited. | Extend cbuffer layout handling and add source-level tests for vectors, matrices, arrays, and nonzero offsets before claiming broader cbuffer support. |
| Global/UAV atomics | RTL global `I_ATOMIC` has opcodes for integer add/and/or/xor/min/max/swap/CAS and also float add/min/max. | Compiler supports integer `atomicrmw`/`cmpxchg` shapes used by OpenCL and HLSL `Interlocked*`. Float atomics and some exact source-level variants are not covered. | Add IR/SelectionDAG coverage for float atomics only if a source language path needs them; otherwise keep them documented as RTL capacity. |
| Command processor operations | RTL/host command path supports DMA, memset, fences, semaphores, engine fences, barriers, dispatch, and workgroup dispatch. | LLVM emits kernel objects and metadata, not command buffers. Runtime/host code owns command submission. | Do not move these into the compiler unless a real command-buffer compiler is introduced. Keep launch metadata in `.gpu.meta` and command construction in the host runtime. |
| Native framebuffer / HDMI scanout | RTL can scan out a native 1920x1080 ARGB8888 framebuffer from DDR; host tests write patterns to that framebuffer. | Compiler only sees ordinary DDR stores. It does not model scanout surfaces, swaps, or display timing. | Not a backend lowering target unless we introduce a graphics/display ABI. For now, kernels should write the agreed DDR framebuffer address through normal memory operations or `PIXEL_OUT`. |
| Texture-like workloads | Doom and software renderers use manually uploaded texture data and ordinary loads. | There is no hardware sampler and no HLSL `Texture2D.Sample` / sampler lowering. | If needed short term, lower restricted texture access as explicit buffer loads in software. Do not present this as native texture/sampler support. |

Near-term compiler priorities from this list:

1. Add simulator coverage for the new `llvm.gpu.pixel.out` path, then decide
   whether to expose it through a source-level HLSL/OpenCL builtin.
2. Tighten and document the resource binding model before adding descriptor
   arrays or dynamic indexing, because the ABI decision affects host metadata,
   HLSL lowering, and SPIR-V resource lowering together.
3. Extend cbuffer layout support only with lit tests that match the actual
   Clang-emitted IR shapes, so we do not recreate the old script-only cbuffer
   rewriting path inside the compiler.

## External Tools

| Tool | Location | Use |
|------|----------|-----|
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
| `GPUSPIRVLowering.cpp` | OpenCL/SPIR-V builtin lowering onto raw workgroup state + compiler-derived IDs; also marks helper functions `alwaysinline + internal` so vector / aggregate-typed helpers disappear before ISel |
| `GPUHLSLLowering.cpp` | HLSL lowering: system values, resources, wave ops, barriers, and `groupshared` intrinsics |
| `GPUSubwordMemoryLowering.cpp` | IR-level rewriter that turns ordinary `addrspace(1)` `i1`/`i8`/`i16` loads/stores into aligned i32 word load + shift/mask (loads) or read-modify-write (stores). Atomic/volatile sub-word global stores are rejected explicitly instead of being mislowered through that RMW path. |
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
  - now runs `GPUSubwordMemoryLowering` (sub-word global mem rewriter),
    `AlwaysInlinerLegacyPass` + `GlobalDCE` (folds non-kernel helpers
    away before ISel — needed for vector-returning helpers like Rodinia
    `mergesort`'s `sortElem(float4)`), and `Scalarizer` (breaks any
    remaining vector ops in the kernel body into scalar i32/f32)
  - reason: the backend cannot emit honest `LD_LOCAL` / `ST_LOCAL` /
    `ATOMIC_LOCAL` if local/shared memory is erased before instruction
    selection, and the host cannot stop hand-filling launch metadata unless the
    object carries a stable reflection record. The hardware also has only
    32-bit memory ops and no multi-register return ABI, so any kernel that
    uses `__global char*` or any helper with vector/aggregate signatures
    must have those rewritten before ISel.

- `llvm/lib/Target/GPU/GPUInstrInfo.td`
  - added machine instruction definitions for `GETSR`, `LD_LOCAL`, `ST_LOCAL`,
    `ATOMIC_LOCAL`, `ATOMIC_LOCAL_CAS`, `BARRIER`, `MEM_FENCE`, `FSQRT`, and
    `HALT_RET`
  - reason: these are the compiler-visible instructions required by the new
    compute ABI and simulator/runtime bring-up
  - reviewer note: `HALT_RET` is not a new hardware opcode; it keeps a return
    register operand alive through lowering while encoding the same `halt`

- `llvm/lib/Target/GPU/MCTargetDesc/GPUMCCodeEmitter.cpp`
  - added actual binary encoding for those instructions, including `FSQRT`
  - reason: instruction defs alone are not enough; the backend still has to
    emit the right 128-bit ISA words

- `llvm/lib/Target/GPU/GPUISelLowering.h`
  - added the `GETSR` target DAG node declaration
  - reason: the backend needs an explicit internal node to carry special-register
    reads from IR intrinsic lowering into instruction selection

- `llvm/lib/Target/GPU/GPUISelLowering.cpp`
  - lowers `llvm.gpu.getsr` to `GPUISD::GETSR`
  - marks `ISD::FSQRT` legal so SelectionDAG can emit the native opcode
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
  - added math-builtin rewriting for OpenCL `fabs`, `mul24`, `mad24`,
    `fma` to `llvm.fabs.f32` / plain `mul` / `mul + add` / `llvm.fma.f32`
    so they reach ISel as ordinary legal IR instead of unhandled libcalls
    that crash `LowerReturn`
  - added `markHelpersAlwaysInline` which tags every non-kernel,
    non-`optnone` helper as `alwaysinline + internal`. Functions with
    no callers and a calling-convention-incompatible signature
    (vector / aggregate / i64 / ...) also get internalized so the
    follow-up GlobalDCE drops them. Functions that look like top-level
    entry points (CC-compatible signature, no callers — e.g. lit-test
    `define void @test_*` cases) are left alone so the lit suite
    survives. Must run before `stripCallingConventions` so the
    `SPIR_KERNEL` calling convention is still visible.
  - reviewer note: this file now explicitly ignores non-declaration functions
    during builtin classification so real functions are not misidentified and
    deleted just because their names contain builtin substrings

- `llvm/lib/Target/GPU/GPUPeephole.cpp` /
  `llvm/lib/Target/GPU/GPUMCInstLower.cpp`
  - source-modifier folding still recognizes
    `FSUB(MOVI(0),x) -> NEG` and `ANDi(x,0x7FFFFFFF) -> ABS`, but the folded
    modifier bits are now carried in a backend-side side table keyed by
    `MachineInstr*` instead of abusing register-operand target flags
  - reason: this LLVM branch asserts if target flags are attached to a
    register `MachineOperand`; the side-table path keeps the same final
    encoding contract without crashing assertion-enabled `llc`

- `llvm/tools/gpu-compiler/CMakeLists.txt`
  - now links IPO explicitly
  - reason: the GPU target machine directly calls
    `createAlwaysInlinerLegacyPass()` and `createGlobalDCEPass()`, so the
    standalone `gpu-compiler` tool must link the IPO library instead of
    relying on transitive linkage that happened to work on some hosts

- `llvm/lib/Target/GPU/GPUHLSLLowering.cpp`
  - moved HLSL system values to raw workgroup state + compiler-derived IDs
  - added lowering for `GroupMemoryBarrierWithGroupSync()`
  - added lowering for `groupshared` `Interlocked*`
  - now rebuilds `rsqrt` from a native `sqrt` intrinsic plus reciprocal instead
    of a hand-rolled refinement sequence
  - updated the `llvm.sqrt` declaration lookup to pass `Module*` instead of
    `Module&`
  - reason: current upstream LLVM in this submodule exposes
    `Intrinsic::getOrInsertDeclaration()` on `Module*`, and the stale call site
    made `Scripts/compiler-build.sh` fail while compiling `GPUHLSLLowering.cpp`
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
  - now also emits the `direct_local_arg_mask` for kernels with `>4`
    args (the indirect-arg path), not just for the direct-register
    `<=4` path. The host needs this to know which slots in the
    indirect arg buffer to pack as `__local` byte offsets vs DDR
    pointers — Rodinia `pathfinder` is the use case (12 args, two
    `__local int*`).

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

- `llvm/lib/Target/GPU/GPUFrameLowering.cpp`
  - the spill-stack base prologue used to be `r30 = lane * frame_size +
    0x00380000`, which gave every concurrently-running workgroup the same
    DDR region per lane. With four engines all running different
    workgroups in parallel, lane 5 of engine 0 and lane 5 of engine 1
    both wrote to the same address.
  - now the prologue reads `group_id_x/y/z` and `num_groups_x/y` via
    `GETSR`, computes the linear group id, and partitions the spill
    region per workgroup:
    `r30 = (linear_gid * 8 + lane) * frame_size + 0x00380000`. Scratch
    is r25..r29 + r30, all unused at function entry.
  - reason: without this, hotspot3D produces `-inf` (and other garbage)
    when dispatched across multiple engines. With it, the four engines
    can run independently from a shared DDR spill area without aliasing.

- `llvm/lib/Target/GPU/GPUSubwordMemoryLowering.cpp`
  - new IR module pass that rewrites every `addrspace(1)` (and
    `addrspace(0)`) `i1`/`i8`/`i16` load and store into an aligned
    32-bit word access. Loads become
    `(load i32 (addr & ~3)) >> ((addr & 3) * 8)` masked to the width;
    stores become a read-modify-write of the surrounding word.
  - reason: the GPU only has `LD_SCATTER` / `ST_SCATTER` 32-bit memory
    ops, so a `__global char*` (Rodinia `bfs`) hits ISel with no
    pattern. This pass is the simplest thing that gets those kernels
    through the compiler.
  - reviewer note: this pass only handles ordinary non-atomic,
    non-volatile stores. Atomic or volatile sub-word global stores are
    rejected explicitly because a load-mask-store rewrite would silently
    miscompile their semantics. addrspace(3) is intentionally left alone
    because the local-memory backend has its own LD_LOCAL/ST_LOCAL path.

- `llvm/lib/Target/GPU/GPUISelLowering.cpp` / `.h`
  - lowers `llvm.gpu.getsr` to `GPUISD::GETSR`
  - adds return-lowering support so return values stay live to `HALT_RET`
  - changes extload action for i1/i8/i16 from `Promote` to `Expand`,
    so `load i32 + and 0xff` is not folded back into an i8 extload that
    the legalizer cannot expand on this target
  - overrides `shouldReduceLoadWidth` to refuse i1/i8/i16 narrowing —
    the GPU only has 32-bit loads, so there is no profit in turning a
    full word load into a sub-word extload
  - calls `setMaxDivRemBitWidthSupported(0)` so the IR-level
    `ExpandIRInsts` pass synthesizes the bit-by-bit shift/subtract
    sequence for `sdiv`/`udiv`/`srem`/`urem` before ISel — the
    hardware has no integer division and the runtime has no
    `__divsi3` libcall
  - reason: special-register reads, sub-word memory shapes, integer
    division, and the updated test coverage all need proper lowering
    through the target DAG

- `llvm/lib/Target/GPU/README.md`
  - updated to reflect the current backend ABI, IR-pass pipeline,
    and source-level status
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

Current GPU lit suite (`37` tests):

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
- `integer-divrem-expand.ll` — variable-divisor `sdiv`/`udiv`/`srem`/`urem`
  expand via `ExpandIRInsts` instead of a missing libcall
- `load-mask-subword.ll` — `(load i32) & 0xff` is not converted to an
  i8 extload, and the same shape works as a `__local` index
- `local-atomic.ll`
- `local-memory.ll`
- `loop-break-divergent.ll`
- `loop.ll`
- `memory.ll`
- `movi-fold.ll`
- `opencl-fabs-builtin.ll` — OpenCL `fabs(float)` libcall is rewritten
  to `llvm.fabs.f32` and folded into a FABS source modifier
- `opencl-local-arg-o0.ll`
- `opencl-local-arg.ll`
- `opencl-local-atomic-builtins.ll`
- `opencl-mul24-builtin.ll` — OpenCL `mul24` / `mad24` libcalls are
  rewritten to plain mul / mul+add
- `opencl-sync-builtins.ll`
- `opencl-workgroup-builtins.ll`
- `reduce.ll`
- `select.ll`
- `signed-int-compare.ll`
- `source-modifiers.ll`
- `spill.ll`
- `subword-global-memory-reject.ll` — atomic/volatile sub-word global
  stores are rejected instead of being lowered through the ordinary
  load-mask-store rewrite
- `subword-global-memory.ll` — `__global char*`/`short*` loads/stores
  for ordinary non-atomic, non-volatile accesses go through
  `GPUSubwordMemoryLowering` (aligned i32 word + shift/mask for loads,
  RMW for stores)
- `sync.ll`
- `uitof-ftou.ll`
- `vector-helper-inline.ll` — non-kernel helpers with vector signatures
  get inlined + DCE'd before ISel by `markHelpersAlwaysInline` +
  `AlwaysInlinerLegacyPass` + `Scalarizer` + `GlobalDCE`

Useful superproject host-side checks alongside the LLVM lit suite:

- `Source/Host/Tests/compiler_verify.c`
- `Source/Host/Tests/gpu_sim_test.c`
- `Source/Host/Tests/break_color_test.c`
- `Source/Host/Tests/rejection_loop_test.c`
- `Source/Host/Tests/gpu_hw_test.c`
- `Source/Host/Tests/Shaders/Benchmarks/build.sh` — compile-only
  sweep of 8 in-house + 12 unmodified Rodinia OpenCL kernels through
  the repo backend (`20/20` green)
- `Source/Host/Tests/Rodinia/run_all.sh` — end-to-end FPGA harness
  for unmodified Rodinia `vec_add`, `hotspot3D`, `gaussian` (Fan1),
  and `kmeans`. Each harness dispatches across all four compute
  engines via separate `OP_DISPATCH_WORKGROUP` commands and verifies
  bit-exactly against a CPU reference; output also includes
  Rodinia-style throughput / GFLOP/s / GB/s reporting from the
  per-engine performance counters.
- `Source/Host/PathTracer/pathtracer_compare.c`

## Pipeline

1. IR-level lowering passes in `GPUTargetMachine::addIRPasses` (in order):
   `GPUSPIRVLowering` (OpenCL builtin lowering, helper functions tagged
   `alwaysinline + internal`), `GPUHLSLLowering`, `GPUSubwordMemoryLowering`
   (`addrspace(1)` i1/i8/i16 ordinary loads/stores → aligned i32 word +
   shift/mask; atomic/volatile sub-word stores rejected),
   `GPULocalMemoryGlobalsPass`, `GPUKernelMetadata`, `AlwaysInlinerLegacyPass`
   + `GlobalDCE` (drops the helper functions whose signatures the GPU
   calling convention can't carry), `Scalarizer` (breaks any remaining
   vector ops into scalar i32/f32), then the standard `FixIrreducible`,
   `UnifyLoopExits`, `StructurizeCFG`, and `SimplifyCFG`.
2. `GPUFrameLowering::emitPrologue` partitions the DDR spill region per
   workgroup using the linear group id read from the wg-context special
   registers, so concurrent workgroups on different engines do not collide.
3. `ExpandIRInsts` (gated by `setMaxDivRemBitWidthSupported(0)`) expands
   integer `sdiv`/`udiv`/`srem`/`urem` into bit-by-bit shift/subtract.
4. Normal instruction selection lowers scalar LLVM IR to GPU machine
   instructions.
5. `GPUControlFlow.cpp` converts structured machine CFG regions into
   `WHILE/BREAK/JUMP/JOIN` and `GOTO/JOIN`.
6. `GPUPeephole.cpp` performs late local combines and patches final branch
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
- full simulator memory-model fidelity beyond the current supported
  workgroup-local barrier model

Hardware limitations and how the backend papers over them:

- no integer division — `setMaxDivRemBitWidthSupported(0)` forces the
  generic `ExpandIRInsts` IR pass to expand all `sdiv`/`udiv`/`srem`/
  `urem` into the bit-by-bit shift/subtract sequence from
  `IntegerDivision.cpp` before ISel.
- native `FSQRT` is available; `FSIN`, `FCOS`, `FPOW`, `FEXP`, and `FLOG`
  still fail. Add polynomial approximations in source where needed.
- no sub-word loads/stores (`LD_SCATTER` / `ST_SCATTER` are 32-bit) —
  the `GPUSubwordMemoryLowering` IR pass rewrites every `i1`/`i8`/`i16`
  load on `addrspace(1)` into an aligned i32 word load + shift/mask,
  and every sub-word store into a (racy across lanes) read-modify-write.
- no vector / aggregate calling convention — `GPUSPIRVLowering` marks
  every non-kernel helper `alwaysinline + internal`, then the IR pass
  pipeline runs `AlwaysInliner` + `Scalarizer` + `GlobalDCE` so vector
  helpers like Rodinia mergesort's `sortElem(float4)` disappear before
  ISel.
- `HALT` terminates all lanes, not individual lanes.

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
