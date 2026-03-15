; RUN: llc -march=gpu < %s | FileCheck %s

define void @test_add(i32 %a, i32 %b, ptr %out) {
; CHECK: add
  %r = add i32 %a, %b
  store i32 %r, ptr %out
  ret void
}

define void @test_add_imm(i32 %a, ptr %out) {
; CHECK: add{{.*}}0x2a
  %r = add i32 %a, 42
  store i32 %r, ptr %out
  ret void
}

define void @test_sub(i32 %a, i32 %b, ptr %out) {
; CHECK: sub
  %r = sub i32 %a, %b
  store i32 %r, ptr %out
  ret void
}

define void @test_mul(i32 %a, i32 %b, ptr %out) {
; CHECK: mul
  %r = mul i32 %a, %b
  store i32 %r, ptr %out
  ret void
}

define void @test_and(i32 %a, i32 %b, ptr %out) {
; CHECK: and
  %r = and i32 %a, %b
  store i32 %r, ptr %out
  ret void
}

define void @test_or(i32 %a, i32 %b, ptr %out) {
; CHECK: or
  %r = or i32 %a, %b
  store i32 %r, ptr %out
  ret void
}

define void @test_xor(i32 %a, i32 %b, ptr %out) {
; CHECK: xor
  %r = xor i32 %a, %b
  store i32 %r, ptr %out
  ret void
}

define void @test_shl(i32 %a, i32 %b, ptr %out) {
; CHECK: shl
  %r = shl i32 %a, %b
  store i32 %r, ptr %out
  ret void
}

define void @test_shr(i32 %a, i32 %b, ptr %out) {
; CHECK: shr
  %r = lshr i32 %a, %b
  store i32 %r, ptr %out
  ret void
}

define void @test_shra(i32 %a, i32 %b, ptr %out) {
; CHECK: shra
  %r = ashr i32 %a, %b
  store i32 %r, ptr %out
  ret void
}

define void @test_neg(i32 %a, ptr %out) {
; CHECK: neg
  %r = sub i32 0, %a
  store i32 %r, ptr %out
  ret void
}
