; RUN: llc -march=gpu %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -march=gpu -print-after=gpu-hlsl-lowering %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=IR

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare float @llvm.dx.fdot.v2f32(<2 x float>, <2 x float>)
declare float @llvm.dx.fdot.v3f32(<3 x float>, <3 x float>)
declare float @llvm.dx.fdot.v4f32(<4 x float>, <4 x float>)

define void @hlsl_fdot(ptr %in, ptr %out) #0 {
; ASM-LABEL: hlsl_fdot:
; ASM: fmul
; ASM: fma
; ASM: fma
; ASM: fma
; ASM: st_scatter
; ASM: halt
;
; IR-LABEL: IR Dump After GPU HLSL Lowering
; IR-NOT: llvm.dx.fdot
; IR: extractelement <2 x float>
; IR: call float @llvm.fma.f32
entry:
  %a0p = getelementptr float, ptr %in, i32 0
  %a0 = load float, ptr %a0p, align 4
  %a1p = getelementptr float, ptr %in, i32 1
  %a1 = load float, ptr %a1p, align 4
  %a2p = getelementptr float, ptr %in, i32 2
  %a2 = load float, ptr %a2p, align 4
  %a3p = getelementptr float, ptr %in, i32 3
  %a3 = load float, ptr %a3p, align 4
  %b0p = getelementptr float, ptr %in, i32 4
  %b0 = load float, ptr %b0p, align 4
  %b1p = getelementptr float, ptr %in, i32 5
  %b1 = load float, ptr %b1p, align 4
  %b2p = getelementptr float, ptr %in, i32 6
  %b2 = load float, ptr %b2p, align 4
  %b3p = getelementptr float, ptr %in, i32 7
  %b3 = load float, ptr %b3p, align 4

  %a20 = insertelement <2 x float> poison, float %a0, i32 0
  %a2v = insertelement <2 x float> %a20, float %a1, i32 1
  %b20 = insertelement <2 x float> poison, float %b0, i32 0
  %b2v = insertelement <2 x float> %b20, float %b1, i32 1
  %dot2 = call float @llvm.dx.fdot.v2f32(<2 x float> %a2v, <2 x float> %b2v)

  %a30 = insertelement <3 x float> poison, float %a0, i32 0
  %a31 = insertelement <3 x float> %a30, float %a1, i32 1
  %a3v = insertelement <3 x float> %a31, float %a2, i32 2
  %b30 = insertelement <3 x float> poison, float %b0, i32 0
  %b31 = insertelement <3 x float> %b30, float %b1, i32 1
  %b3v = insertelement <3 x float> %b31, float %b2, i32 2
  %dot3 = call float @llvm.dx.fdot.v3f32(<3 x float> %a3v, <3 x float> %b3v)

  %a40 = insertelement <4 x float> poison, float %a0, i32 0
  %a41 = insertelement <4 x float> %a40, float %a1, i32 1
  %a42 = insertelement <4 x float> %a41, float %a2, i32 2
  %a4v = insertelement <4 x float> %a42, float %a3, i32 3
  %b40 = insertelement <4 x float> poison, float %b0, i32 0
  %b41 = insertelement <4 x float> %b40, float %b1, i32 1
  %b42 = insertelement <4 x float> %b41, float %b2, i32 2
  %b4v = insertelement <4 x float> %b42, float %b3, i32 3
  %dot4 = call float @llvm.dx.fdot.v4f32(<4 x float> %a4v, <4 x float> %b4v)

  %sum0 = fadd float %dot2, %dot3
  %sum1 = fadd float %sum0, %dot4
  store float %sum1, ptr %out, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }
