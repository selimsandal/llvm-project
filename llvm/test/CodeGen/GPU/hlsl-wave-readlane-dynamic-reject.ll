; RUN: not llc -march=gpu %s -o /dev/null 2>&1 | FileCheck %s

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare i32 @llvm.dx.wave.readlane.i32(i32, i32)

define void @hlsl_wave_readlane_dynamic(i32 %x, i32 %lane, ptr %out) #0 {
; CHECK: error: {{.*}}GPU HLSL lowering only supports WaveReadLaneAt with a constant lane index
entry:
  %r = call i32 @llvm.dx.wave.readlane.i32(i32 %x, i32 %lane)
  store i32 %r, ptr %out, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }
