; RUN: llc -march=gpu -filetype=obj %s -o %t.o
; RUN: llc -march=gpu %s -o - | FileCheck %s

; Test: clang-style HLSL cbuffer member globals lower to GPU memory loads.
;
; Equivalent HLSL:
;   cbuffer Params : register(b0) {
;     float scale;
;     uint offset;
;   };
;   RWStructuredBuffer<float> outBuf : register(u1);
;
;   [numthreads(8, 1, 1)]
;   void CSMain(uint3 tid : SV_DispatchThreadID) {
;     outBuf[tid.x] = scale + (float)offset;
;   }

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p2:32:32-p3:32:32-i32:32-f32:32-n32"

%__cblayout_Params = type <{ float, i32 }>

@Params.cb = local_unnamed_addr global target("dx.CBuffer", %__cblayout_Params) poison
@scale = external hidden local_unnamed_addr addrspace(2) global float, align 4
@offset = external hidden local_unnamed_addr addrspace(2) global i32, align 4
@Params.str = private unnamed_addr constant [7 x i8] c"Params\00", align 1
@out.str = private unnamed_addr constant [7 x i8] c"outBuf\00", align 1

declare target("dx.CBuffer", %__cblayout_Params) @llvm.dx.resource.handlefrombinding.tdx.CBuffer_s___cblayout_Paramsst(i32, i32, i32, i32, ptr)
declare target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32, i32, i32, i32, ptr)
declare i32 @llvm.dx.thread.id(i32)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t.i32(target("dx.RawBuffer", float, 1, 0), i32)

define void @main() #0 {
entry:
  %params = call target("dx.CBuffer", %__cblayout_Params) @llvm.dx.resource.handlefrombinding.tdx.CBuffer_s___cblayout_Paramsst(i32 0, i32 0, i32 1, i32 0, ptr @Params.str)
  store target("dx.CBuffer", %__cblayout_Params) %params, ptr @Params.cb, align 4

  %out_handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 1, i32 1, i32 0, ptr @out.str)
  %id = call i32 @llvm.dx.thread.id(i32 0)

  %scale = load float, ptr addrspace(2) @scale, align 4
  %offset = load i32, ptr addrspace(2) @offset, align 4
  %offset_f = uitofp i32 %offset to float
  %result = fadd float %scale, %offset_f

  %out_ptr = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t.i32(target("dx.RawBuffer", float, 1, 0) %out_handle, i32 %id)
  store float %result, ptr %out_ptr, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }

!hlsl.cbs = !{!0}
!0 = !{ptr @Params.cb, ptr addrspace(2) @scale, ptr addrspace(2) @offset}

; CHECK: ld_scatter
; CHECK: ld_scatter
; CHECK: uitof
; CHECK: fadd
; CHECK: st_scatter
; CHECK: halt
