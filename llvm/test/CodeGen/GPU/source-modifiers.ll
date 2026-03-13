; RUN: llc -march=gpu < %s | FileCheck %s

; fneg feeding fadd: should fold into src_mod, no separate fsub
define float @test_fneg_fold(float %a, float %b) {
; CHECK: fadd
; CHECK-NOT: fsub
  %neg = fneg float %b
  %r = fadd float %a, %neg
  ret float %r
}

; fabs feeding fmul: should fold into src_mod, no separate and
define float @test_fabs_fold(float %a, float %b) {
; CHECK: fmul
; CHECK-NOT: and
  %abs = call float @llvm.fabs.f32(float %b)
  %r = fmul float %a, %abs
  ret float %r
}

declare float @llvm.fabs.f32(float)
