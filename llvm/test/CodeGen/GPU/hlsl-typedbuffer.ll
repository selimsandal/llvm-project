; RUN: llc -march=gpu -filetype=obj %s -o %t.o
; RUN: llc -march=gpu %s -o - | FileCheck %s

; Test: HLSL TypedBuffer load/store lowering.
;
; Equivalent HLSL shape:
;   Buffer<float> input : register(t0);
;   RWBuffer<float> output : register(u1);
;
;   [numthreads(8, 1, 1)]
;   void CSMain(uint id : SV_DispatchThreadID) {
;       output[id] = input[id] + 1.0f;
;   }

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare i32 @llvm.dx.thread.id(i32)

declare target("dx.TypedBuffer", float, 0, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_0_0_0t(i32, i32, i32, i32, ptr)
declare target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32, i32, i32, i32, ptr)
declare { float, i1 } @llvm.dx.resource.load.typedbuffer.f32.tdx.TypedBuffer_f32_0_0_0t(target("dx.TypedBuffer", float, 0, 0, 0), i32)
declare void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_f32_1_0_0t.f32(target("dx.TypedBuffer", float, 1, 0, 0), i32, float)

@input.str = private unnamed_addr constant [6 x i8] c"input\00"
@output.str = private unnamed_addr constant [7 x i8] c"output\00"

define void @main() #0 {
entry:
  %id = call i32 @llvm.dx.thread.id(i32 0)

  %input_handle = call target("dx.TypedBuffer", float, 0, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_0_0_0t(i32 0, i32 0, i32 1, i32 0, ptr @input.str)
  %output_handle = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 1, i32 1, i32 0, ptr @output.str)

  %load = call { float, i1 } @llvm.dx.resource.load.typedbuffer.f32.tdx.TypedBuffer_f32_0_0_0t(target("dx.TypedBuffer", float, 0, 0, 0) %input_handle, i32 %id)
  %value = extractvalue { float, i1 } %load, 0
  %result = fadd float %value, 1.000000e+00
  call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_f32_1_0_0t.f32(target("dx.TypedBuffer", float, 1, 0, 0) %output_handle, i32 %id, float %result)
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }

; CHECK: ld_scatter
; CHECK: fadd
; CHECK: st_scatter
; CHECK: halt
