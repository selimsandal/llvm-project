; RUN: llc -march=gpu %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -march=gpu -print-after=gpu-hlsl-lowering %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=IR

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

%"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0), target("dx.RawBuffer", float, 1, 0) }

@Out = internal global %"class.hlsl::RWStructuredBuffer" poison, align 4
@Out.str = private unnamed_addr constant [4 x i8] c"Out\00", align 1

declare target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32, i32, i32, i32, ptr)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t.i32(target("dx.RawBuffer", float, 1, 0), i32)

define void @hlsl_o0_resource_wrapper() #0 {
; ASM-LABEL: hlsl_o0_resource_wrapper:
; ASM: st_scatter
; ASM: halt
;
; IR-LABEL: define void @hlsl_o0_resource_wrapper
; IR-NOT: target("dx.RawBuffer"
; IR-NOT: llvm.dx.resource
; IR: store float
entry:
  %tmp = alloca %"class.hlsl::RWStructuredBuffer", align 4
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @Out.str)
  store target("dx.RawBuffer", float, 1, 0) %handle, ptr %tmp, align 4
  %tmp.handle = load target("dx.RawBuffer", float, 1, 0), ptr %tmp, align 4
  store target("dx.RawBuffer", float, 1, 0) %tmp.handle, ptr @Out, align 4
  %out.handle = load target("dx.RawBuffer", float, 1, 0), ptr @Out, align 4
  %ptr = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t.i32(target("dx.RawBuffer", float, 1, 0) %out.handle, i32 0)
  store float 1.000000e+00, ptr %ptr, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }
