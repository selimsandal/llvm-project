; RUN: llc -march=gpu < %s | FileCheck %s

; Regression for: OpenCL `fabs(float)` used to reach llc as a declared
; libcall to `_Z4fabsf`, which crashed `GPUTargetLowering::LowerReturn`
; with a `std::length_error` during call/return lowering. GPUSPIRVLowering
; now recognizes the mangled name and rewrites it to `llvm.fabs.f32`,
; which then Expands to `and x, 0x7fffffff` and is folded by GPUPeephole
; into a proper FABS source modifier.

target triple = "spir"

declare dso_local spir_func float @_Z4fabsf(float) local_unnamed_addr

define dso_local spir_kernel void @fabs_test(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: fabs_test:
; CHECK:       ld_scatter
; CHECK:       {{fabs|and.*0x7fffffff}}
; CHECK:       st_scatter
; CHECK:       halt
  %v = load float, ptr addrspace(1) %in, align 4
  %r = call spir_func float @_Z4fabsf(float %v)
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}
