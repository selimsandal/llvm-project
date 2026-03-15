; RUN: llc -march=gpu < %s | FileCheck %s

; fneg feeding fadd: DAG-combined to fsub, no separate fneg instruction
define void @test_fneg_fold(float %a, float %b, ptr %out) {
; CHECK-LABEL: test_fneg_fold:
; CHECK: fsub
; CHECK-NOT: movi
  %neg = fneg float %b
  %r = fadd float %a, %neg
  store float %r, ptr %out
  ret void
}

; fabs feeding fmul: should fold into src_mod, no separate and
define void @test_fabs_fold(float %a, float %b, ptr %out) {
; CHECK-LABEL: test_fabs_fold:
; CHECK: fmul
; CHECK-NOT: and
  %abs = call float @llvm.fabs.f32(float %b)
  %r = fmul float %a, %abs
  store float %r, ptr %out
  ret void
}

declare float @llvm.fabs.f32(float)
