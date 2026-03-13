; RUN: llc -march=gpu < %s | FileCheck %s

; Integer compare + select
define i32 @test_select_lt(i32 %a, i32 %b, i32 %t, i32 %f) {
; CHECK: cmp
; CHECK: sel
  %cmp = icmp slt i32 %a, %b
  %r = select i1 %cmp, i32 %t, i32 %f
  ret i32 %r
}

; Unsigned compare + select
define i32 @test_select_ult(i32 %a, i32 %b, i32 %t, i32 %f) {
; CHECK: cmp
; CHECK: sel
  %cmp = icmp ult i32 %a, %b
  %r = select i1 %cmp, i32 %t, i32 %f
  ret i32 %r
}

; Float compare + select
define float @test_select_flt(float %a, float %b, float %t, float %f) {
; CHECK: cmp
; CHECK: sel
  %cmp = fcmp olt float %a, %b
  %r = select i1 %cmp, float %t, float %f
  ret float %r
}

; Equality compare
define i32 @test_select_eq(i32 %a, i32 %b, i32 %t, i32 %f) {
; CHECK: cmp
; CHECK: sel
  %cmp = icmp eq i32 %a, %b
  %r = select i1 %cmp, i32 %t, i32 %f
  ret i32 %r
}
