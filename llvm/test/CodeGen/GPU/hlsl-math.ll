; RUN: llc -march=gpu %s -o - | FileCheck %s

; Test: HLSL math intrinsic lowering (lerp, saturate, step)

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32, i32, i32, i32, ptr)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0), i32)
declare i32 @llvm.dx.thread.id(i32)
declare float @llvm.dx.lerp.f32(float, float, float)
declare float @llvm.dx.saturate.f32(float)
declare float @llvm.dx.step.f32(float, float)

@buf.str = private unnamed_addr constant [4 x i8] c"buf\00"
@out.str = private unnamed_addr constant [4 x i8] c"out\00"

define void @main() #0 {
entry:
  %id = call i32 @llvm.dx.thread.id(i32 0)

  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @buf.str)
  %handle_out = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 1, i32 1, i32 0, ptr @out.str)

  ; Load buf[id]
  %ptr_rd = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %handle, i32 %id)
  %x = load float, ptr %ptr_rd

  ; lerp(x, x+1.0, 0.5) — non-trivial lerp to prevent constant folding
  %x1 = fadd float %x, 1.0
  %lr = call float @llvm.dx.lerp.f32(float %x, float %x1, float 0.5)

  ; saturate(result)
  %sat = call float @llvm.dx.saturate.f32(float %lr)

  ; step(0.5, sat)
  %st = call float @llvm.dx.step.f32(float 0.5, float %sat)

  ; Store back
  %ptr_wr = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %handle_out, i32 %id)
  store float %st, ptr %ptr_wr

  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }

; CHECK: ld_scatter
; lerp uses fma
; CHECK: fma
; saturate uses fmax then fmin
; CHECK: fmin
; step uses cmp + sel
; CHECK: cmp
; CHECK: sel
; CHECK: st_scatter
; CHECK: halt
