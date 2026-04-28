; RUN: llc -march=gpu %s -o - | FileCheck %s
; RUN: llc -march=gpu -print-after=gpu-hlsl-lowering %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=IR

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare float @llvm.dx.frac.f32(float)

define void @hlsl_frac(ptr %in, ptr %out) #0 {
; CHECK-LABEL: hlsl_frac:
; CHECK: ftoi
; CHECK: itof
; CHECK: cmp
; CHECK: sel
; CHECK: st_scatter
; CHECK: halt
;
; IR-LABEL: IR Dump After GPU HLSL Lowering
; IR-NOT: llvm.dx.frac
; IR: fcmp olt float
; IR: select i1
entry:
  %x = load float, ptr %in, align 4
  %frac = call float @llvm.dx.frac.f32(float %x)
  store float %frac, ptr %out, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }
