; RUN: llc -march=gpu < %s | FileCheck %s

define void @test_add(i32 %a, i32 %b) {
; CHECK: add
  %r = add i32 %a, %b
  ret void
}

define void @test_add_imm(i32 %a) {
; CHECK: add{{.*}}0x2a
  %r = add i32 %a, 42
  ret void
}

define void @test_sub(i32 %a, i32 %b) {
; CHECK: sub
  %r = sub i32 %a, %b
  ret void
}

define void @test_mul(i32 %a, i32 %b) {
; CHECK: mul
  %r = mul i32 %a, %b
  ret void
}

define void @test_and(i32 %a, i32 %b) {
; CHECK: and
  %r = and i32 %a, %b
  ret void
}

define void @test_or(i32 %a, i32 %b) {
; CHECK: or
  %r = or i32 %a, %b
  ret void
}

define void @test_xor(i32 %a, i32 %b) {
; CHECK: xor
  %r = xor i32 %a, %b
  ret void
}

define void @test_shl(i32 %a, i32 %b) {
; CHECK: shl
  %r = shl i32 %a, %b
  ret void
}

define void @test_shr(i32 %a, i32 %b) {
; CHECK: shr
  %r = lshr i32 %a, %b
  ret void
}

define void @test_shra(i32 %a, i32 %b) {
; CHECK: shra
  %r = ashr i32 %a, %b
  ret void
}

define void @test_neg(i32 %a) {
; CHECK: neg
  %r = sub i32 0, %a
  ret void
}
