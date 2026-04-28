; RUN: not llc -march=gpu %s -o /dev/null 2>&1 | FileCheck %s

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare i32 @llvm.dx.wave.readlane.i32(i32, i32)

define void @hlsl_wave_readlane_lane8(i32 %x, ptr %out) #0 {
; CHECK: error: {{.*}}GPU HLSL lowering requires WaveReadLaneAt constant lane index to be in the fixed 8-lane wave range [0, 7]
entry:
  %r = call i32 @llvm.dx.wave.readlane.i32(i32 %x, i32 8)
  store i32 %r, ptr %out, align 4
  ret void
}

define void @hlsl_wave_readlane_lanem1(i32 %x, ptr %out) #0 {
; CHECK: error: {{.*}}GPU HLSL lowering requires WaveReadLaneAt constant lane index to be in the fixed 8-lane wave range [0, 7]
entry:
  %r = call i32 @llvm.dx.wave.readlane.i32(i32 %x, i32 -1)
  store i32 %r, ptr %out, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }
