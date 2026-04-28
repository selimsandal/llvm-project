; RUN: llc -march=gpu %s -o - | FileCheck %s
; RUN: llc -march=gpu -print-after=gpu-hlsl-lowering %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=IR

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare float @llvm.floor.f32(float)
declare float @llvm.ceil.f32(float)
declare float @llvm.trunc.f32(float)
declare float @llvm.round.f32(float)
declare float @llvm.roundeven.f32(float)
declare float @llvm.sqrt.f32(float)
declare float @llvm.fabs.f32(float)
declare i32 @llvm.abs.i32(i32, i1 immarg)

define void @hlsl_scalar_llvm_math(ptr %inf, ptr %ini, ptr %outf, ptr %outi) #0 {
; CHECK-LABEL: hlsl_scalar_llvm_math:
; CHECK: ftoi
; CHECK: itof
; CHECK: cmp
; CHECK: sel
; CHECK: fsqrt
; CHECK: st_scatter
; CHECK: st_scatter
; CHECK: halt
;
; IR-LABEL: IR Dump After GPU HLSL Lowering
; IR-NOT: llvm.floor
; IR-NOT: llvm.ceil
; IR-NOT: llvm.trunc
; IR-NOT: llvm.round
; IR-NOT: llvm.abs
; IR: fptosi
; IR: sitofp
; IR: fcmp olt float
; IR: fcmp ogt float
; IR: and i32
entry:
  %x = load float, ptr %inf, align 4
  %floor = call float @llvm.floor.f32(float %x)
  %ceil = call float @llvm.ceil.f32(float %x)
  %trunc = call float @llvm.trunc.f32(float %x)
  %round = call float @llvm.round.f32(float %x)
  %roundeven = call float @llvm.roundeven.f32(float %x)
  %sqrt = call float @llvm.sqrt.f32(float %x)
  %fabs = call float @llvm.fabs.f32(float %x)
  %sum0 = fadd float %floor, %ceil
  %sum1 = fadd float %sum0, %trunc
  %sum2 = fadd float %sum1, %round
  %sum3 = fadd float %sum2, %roundeven
  %sum4 = fadd float %sum3, %sqrt
  %sum5 = fadd float %sum4, %fabs
  store float %sum5, ptr %outf, align 4

  %i = load i32, ptr %ini, align 4
  %abs = call i32 @llvm.abs.i32(i32 %i, i1 false)
  store i32 %abs, ptr %outi, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }
