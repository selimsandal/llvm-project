; RUN: llc -march=gpu < %s | FileCheck %s

declare i32 @llvm.gpu.reduce.add(i32)
declare i32 @llvm.gpu.reduce.and(i32)
declare i32 @llvm.gpu.reduce.xor(i32)
declare i32 @llvm.gpu.reduce.smax(i32)
declare float @llvm.gpu.reduce.fadd(float)

define void @test_reduce_add(i32 %x, ptr %out) {
; CHECK: reduce
; CHECK: st_scatter
  %r = call i32 @llvm.gpu.reduce.add(i32 %x)
  store i32 %r, ptr %out
  ret void
}

define void @test_reduce_and(i32 %x, ptr %out) {
; CHECK: reduce
; CHECK: st_scatter
  %r = call i32 @llvm.gpu.reduce.and(i32 %x)
  store i32 %r, ptr %out
  ret void
}

define void @test_reduce_xor(i32 %x, ptr %out) {
; CHECK: reduce
  %r = call i32 @llvm.gpu.reduce.xor(i32 %x)
  store i32 %r, ptr %out
  ret void
}

define void @test_reduce_smax(i32 %x, ptr %out) {
; CHECK: reduce
  %r = call i32 @llvm.gpu.reduce.smax(i32 %x)
  store i32 %r, ptr %out
  ret void
}

define void @test_reduce_fadd(float %x, ptr %out) {
; CHECK: reduce
  %r = call float @llvm.gpu.reduce.fadd(float %x)
  store float %r, ptr %out
  ret void
}
