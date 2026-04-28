; RUN: llc -march=gpu %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -march=gpu -print-after=gpu-hlsl-lowering %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=IR

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare i32 @llvm.dx.wave.readlane.i32(i32, i32)
declare float @llvm.dx.wave.readlane.f32(float, i32)
declare i1 @llvm.dx.wave.readlane.i1(i1, i32)

define void @hlsl_wave_readlane(ptr %ini, ptr %inf, ptr %outi, ptr %outf, ptr %outb) #0 {
; ASM-LABEL: hlsl_wave_readlane:
; ASM: cmp
; ASM: sel
; ASM: reduce
; ASM: reduce
; ASM: reduce
; ASM: st_scatter
; ASM: halt
;
; IR-LABEL: define void @hlsl_wave_readlane
; IR-NOT: llvm.dx.wave.readlane
; IR: call i32 @llvm.gpu.reduce.or
entry:
  %i = load i32, ptr %ini, align 4
  %f = load float, ptr %inf, align 4
  %pred = icmp sgt i32 %i, 0

  %ri = call i32 @llvm.dx.wave.readlane.i32(i32 %i, i32 0)
  %rf = call float @llvm.dx.wave.readlane.f32(float %f, i32 3)
  %rb = call i1 @llvm.dx.wave.readlane.i1(i1 %pred, i32 7)

  store i32 %ri, ptr %outi, align 4
  store float %rf, ptr %outf, align 4
  %rb.i32 = zext i1 %rb to i32
  store i32 %rb.i32, ptr %outb, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }
