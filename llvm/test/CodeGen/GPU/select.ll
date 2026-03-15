; RUN: llc -march=gpu < %s | FileCheck %s

; Integer signed min (icmp slt + select of same operands → smin)
define void @test_select_lt(i32 %a, i32 %b, ptr %out) {
; CHECK: smin
  %cmp = icmp slt i32 %a, %b
  %r = select i1 %cmp, i32 %a, i32 %b
  store i32 %r, ptr %out
  ret void
}

; Integer unsigned min (icmp ult + select of same operands → umin)
define void @test_select_ult(i32 %a, i32 %b, ptr %out) {
; CHECK: umin
  %cmp = icmp ult i32 %a, %b
  %r = select i1 %cmp, i32 %a, i32 %b
  store i32 %r, ptr %out
  ret void
}

; Float compare + select (different true/false values)
define void @test_select_flt(float %a, float %b, ptr %out) {
; CHECK: cmp
; CHECK: sel
  %cmp = fcmp olt float %a, %b
  %r = select i1 %cmp, float %a, float %b
  store float %r, ptr %out
  ret void
}

; Equality compare + select
define void @test_select_eq(i32 %a, i32 %b, ptr %out) {
; CHECK: cmp
; CHECK: sel
  %cmp = icmp eq i32 %a, %b
  %r = select i1 %cmp, i32 %a, i32 %b
  store i32 %r, ptr %out
  ret void
}
