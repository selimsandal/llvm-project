; RUN: llc -march=gpu %s -o - | FileCheck %s

; Test: HLSL wave intrinsic lowering → GPU REDUCE instruction
;
; Equivalent HLSL:
;   RWStructuredBuffer<int> buf : register(u0);
;   [numthreads(8, 1, 1)]
;   void CSMain(uint id : SV_DispatchThreadID) {
;       int val = buf[id];
;       int sum = WaveActiveSum(val);
;       buf[id] = sum;
;   }

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32, i32, i32, i32, ptr)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0), i32)
declare i32 @llvm.dx.thread.id(i32)
declare i32 @llvm.dx.wave.reduce.sum.i32(i32)
declare i32 @llvm.dx.wave.getlaneindex()

@buf.str = private unnamed_addr constant [4 x i8] c"buf\00"

define void @main() #0 {
entry:
  %id = call i32 @llvm.dx.thread.id(i32 0)
  %lane = call i32 @llvm.dx.wave.getlaneindex()

  %handle = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @buf.str)

  ; Load buf[id]
  %ptr_rd = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %handle, i32 %id)
  %val = load i32, ptr %ptr_rd

  ; WaveActiveSum(val) → reduce add
  %sum = call i32 @llvm.dx.wave.reduce.sum.i32(i32 %val)

  ; Store sum to buf[id]
  %ptr_wr = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %handle, i32 %id)
  store i32 %sum, ptr %ptr_wr

  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }

; CHECK: ld_scatter
; CHECK: reduce
; CHECK: st_scatter
; CHECK: halt
