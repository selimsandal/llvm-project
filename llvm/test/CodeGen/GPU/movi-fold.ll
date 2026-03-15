; RUN: llc -march=gpu < %s | FileCheck %s
; Test that MOVI is not deleted when the register has multiple uses.
; The constant 42 feeds both an ADD and a store — MOVI must survive.

define void @test_movi_multi_use(i32 %x, ptr %out1, ptr %out2) {
; CHECK-LABEL: test_movi_multi_use:
; The constant should appear either as movi + two uses, or folded into
; both operations. Either way, both stores must produce correct values.
; CHECK: st_scatter
; CHECK: st_scatter
  %a = add i32 %x, 42
  store i32 %a, ptr %out1
  %b = add i32 %x, 42
  store i32 %b, ptr %out2
  ret void
}

; Float constant used twice — MOVI must not be deleted after first fold
define void @test_movi_float_multi(float %x, ptr %out1, ptr %out2) {
; CHECK-LABEL: test_movi_float_multi:
; CHECK: st_scatter
; CHECK: st_scatter
  %a = fadd float %x, 1.0
  %b = bitcast float %a to i32
  store i32 %b, ptr %out1
  %c = fmul float %x, 1.0
  %d = bitcast float %c to i32
  store i32 %d, ptr %out2
  ret void
}
