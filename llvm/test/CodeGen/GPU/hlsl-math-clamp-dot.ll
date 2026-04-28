; RUN: llc -march=gpu %s -o - | FileCheck %s

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare float @llvm.dx.nclamp.f32(float, float, float)
declare i32 @llvm.dx.sclamp.i32(i32, i32, i32)
declare i32 @llvm.dx.uclamp.i32(i32, i32, i32)
declare float @llvm.dx.dot2.f32(float, float, float, float)
declare float @llvm.dx.dot3.f32(float, float, float, float, float, float)
declare float @llvm.dx.dot4.f32(float, float, float, float, float, float, float, float)

define void @hlsl_clamp_dot(ptr %in, ptr %outf, ptr %outi) #0 {
; CHECK-LABEL: hlsl_clamp_dot:
; CHECK: fmax
; CHECK: fmin
; CHECK: fma
; CHECK: fma
; CHECK: fma
; CHECK-DAG: smax
; CHECK-DAG: smin
; CHECK-DAG: umax
; CHECK-DAG: umin
; CHECK: st_scatter
; CHECK: halt
entry:
  %f0 = load float, ptr %in, align 4
  %in1 = getelementptr float, ptr %in, i32 1
  %f1 = load float, ptr %in1, align 4
  %in2 = getelementptr float, ptr %in, i32 2
  %f2 = load float, ptr %in2, align 4
  %in3 = getelementptr float, ptr %in, i32 3
  %f3 = load float, ptr %in3, align 4

  %clamped = call float @llvm.dx.nclamp.f32(float %f0, float 0.0, float 1.0)
  %dot2 = call float @llvm.dx.dot2.f32(float %f0, float %f1, float %f2, float %f3)
  %dot3 = call float @llvm.dx.dot3.f32(float %f0, float %f1, float %f2, float %f3, float %clamped, float %dot2)
  %dot4 = call float @llvm.dx.dot4.f32(float %f0, float %f1, float %f2, float %f3, float %clamped, float %dot2, float %dot3, float 1.0)
  %fsum0 = fadd float %clamped, %dot2
  %fsum1 = fadd float %fsum0, %dot3
  %fsum2 = fadd float %fsum1, %dot4
  store float %fsum2, ptr %outf, align 4

  %i0 = bitcast float %f0 to i32
  %sclamped = call i32 @llvm.dx.sclamp.i32(i32 %i0, i32 -7, i32 7)
  %uclamped = call i32 @llvm.dx.uclamp.i32(i32 %i0, i32 3, i32 11)
  %isum = add i32 %sclamped, %uclamped
  store i32 %isum, ptr %outi, align 4

  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }
