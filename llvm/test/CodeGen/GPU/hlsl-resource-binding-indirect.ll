; RUN: llc -march=gpu -filetype=obj %s -o %t.o
; RUN: llc -march=gpu -print-after=gpu-hlsl-lowering %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=IR
; RUN: llc -march=gpu %s -o - | FileCheck %s --check-prefix=ASM

; Documents the current indirect HLSL binding model:
; if any resource uses binding slot 4 or above, resource bases are loaded from
; the indirect args buffer whose base pointer is in r1. Slot N is loaded from
; [r1 + N*4], including lower-numbered slots in the same module.

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

@u0.str = private unnamed_addr constant [3 x i8] c"u0\00"
@u4.str = private unnamed_addr constant [3 x i8] c"u4\00"

declare i32 @llvm.dx.thread.id(i32)
declare target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32, i32, i32, i32, ptr)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0), i32)

define void @slot4_forces_indirect_for_all_bindings() #0 {
; IR-LABEL: define void @slot4_forces_indirect_for_all_bindings
; IR: call i32 @llvm.read_register.i32(metadata [[R1:![0-9]+]])
; IR: add i32 {{%[0-9]+}}, 0
; IR: load i32, ptr {{%[A-Za-z0-9_.]+}}
; IR: call i32 @llvm.read_register.i32(metadata [[R1]])
; IR: add i32 {{%[0-9]+}}, 16
; IR: load i32, ptr {{%[A-Za-z0-9_.]+}}
;
; ASM-LABEL: slot4_forces_indirect_for_all_bindings:
; ASM: ld_scatter {{r[0-9]+}}, [r1 + 0x0]
; ASM: ld_scatter {{r[0-9]+}}, [r1 + 0x10]
entry:
  %id = call i32 @llvm.dx.thread.id(i32 0)
  %h0 = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @u0.str)
  %h4 = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 4, i32 1, i32 0, ptr @u4.str)
  %p0 = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %h0, i32 %id)
  %v0 = load i32, ptr %p0, align 4
  %p4 = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %h4, i32 %id)
  store i32 %v0, ptr %p4, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }

; IR-DAG: [[R1]] = !{!"r1"}
