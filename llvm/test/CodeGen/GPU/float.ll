; RUN: llc -march=gpu < %s | FileCheck %s

define void @test_fadd(float %a, float %b) {
; CHECK: fadd
  %r = fadd float %a, %b
  ret void
}

define void @test_fmul(float %a, float %b) {
; CHECK: fmul
  %r = fmul float %a, %b
  ret void
}

define void @test_fsub(float %a, float %b) {
; CHECK: fsub
  %r = fsub float %a, %b
  ret void
}

define void @test_fdiv(float %a, float %b) {
; CHECK: fdiv
  %r = fdiv float %a, %b
  ret void
}

define void @test_itof(i32 %a) {
; CHECK: itof
  %r = sitofp i32 %a to float
  ret void
}

define void @test_ftoi(float %a) {
; CHECK: ftoi
  %r = fptosi float %a to i32
  ret void
}

; FMA formation: fmul + fadd -> fma
define void @test_fma(float %a, float %b, float %c) {
; CHECK: fma
  %t = fmul float %a, %b
  %r = fadd float %t, %c
  ret void
}
