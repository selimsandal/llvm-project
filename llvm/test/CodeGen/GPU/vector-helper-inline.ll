; RUN: llc -march=gpu < %s | FileCheck %s

; Regression for: a non-kernel helper that takes or returns a vector
; type used to crash llc with "unable to allocate function return #1"
; out of CCState::AnalyzeReturn — the GPU calling convention has no
; multi-register return slot. The IR pass pipeline now marks all
; non-kernel helpers `alwaysinline` + `internal`, runs the
; AlwaysInliner, runs the Scalarizer, and finishes with GlobalDCE so
; the float4-returning helper disappears completely before ISel.
;
; This test mirrors what Rodinia hybridsort/mergesort does with the
; `sortElem(float4)` helper.

target triple = "spir"

define dso_local spir_func <4 x float> @vec_helper(<4 x float> %v) {
  %a = extractelement <4 x float> %v, i32 0
  %b = extractelement <4 x float> %v, i32 1
  %c = fadd float %a, %b
  %r0 = insertelement <4 x float> %v, float %c, i32 0
  ret <4 x float> %r0
}

define dso_local spir_kernel void @vec_caller(ptr addrspace(1) %in,
                                               ptr addrspace(1) %out) {
; The kernel must compile, the helper must be inlined and DCE'd, and
; the resulting code must contain real fadd / store instructions.
; CHECK-LABEL: vec_caller:
; CHECK:       fadd
; CHECK:       st_scatter
; CHECK:       halt
; The helper itself must be gone from the final asm.
; CHECK-NOT: vec_helper:
  %p0 = getelementptr inbounds <4 x float>, ptr addrspace(1) %in, i32 0
  %v  = load <4 x float>, ptr addrspace(1) %p0, align 16
  %r  = call spir_func <4 x float> @vec_helper(<4 x float> %v)
  %p1 = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 0
  store <4 x float> %r, ptr addrspace(1) %p1, align 16
  ret void
}
