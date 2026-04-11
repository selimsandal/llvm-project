; RUN: llc -march=gpu < %s | FileCheck %s

; Regression for: a `__global char*` (or `short*`) load/store used to
; reach llc as a direct `load i8` / `store i8` from `addrspace(1)`,
; which crashed ISel with "Cannot select: load (s8) ... anyext from i8"
; — the GPU only has 32-bit memory ops and there is no pattern for
; sub-word loads. The fix is the new `GPUSubwordMemoryLowering` IR
; pass, which rewrites every sub-word global load/store into an
; aligned i32 word load + shift + mask (loads) or read-modify-write
; (stores). This is what unblocks Rodinia bfs, which uses
; `__global char* g_graph_mask` for its visited / mask arrays.

target triple = "spir"

define dso_local spir_kernel void @i8_load(ptr addrspace(1) %p,
                                           ptr addrspace(1) %out) {
; The backend must NOT see a sub-word load — only an aligned i32 load
; followed by shift+mask down to the byte.
; CHECK-LABEL: i8_load:
; CHECK:       ld_scatter
; CHECK:       halt
  %v = load i8, ptr addrspace(1) %p, align 1
  %z = zext i8 %v to i32
  store i32 %z, ptr addrspace(1) %out, align 4
  ret void
}

define dso_local spir_kernel void @i8_store(ptr addrspace(1) %p,
                                            ptr addrspace(1) %vp) {
; CHECK-LABEL: i8_store:
; CHECK:       ld_scatter
; CHECK:       st_scatter
; CHECK:       halt
  %v = load i32, ptr addrspace(1) %vp, align 4
  %t = trunc i32 %v to i8
  store i8 %t, ptr addrspace(1) %p, align 1
  ret void
}

define dso_local spir_kernel void @i16_load(ptr addrspace(1) %p,
                                            ptr addrspace(1) %out) {
; CHECK-LABEL: i16_load:
; CHECK:       ld_scatter
; CHECK:       halt
  %v = load i16, ptr addrspace(1) %p, align 2
  %z = zext i16 %v to i32
  store i32 %z, ptr addrspace(1) %out, align 4
  ret void
}
