; RUN: llc -march=gpu -filetype=obj %s -o %t.o
; RUN: llc -march=gpu -print-after=gpu-hlsl-lowering %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=IR
; RUN: llc -march=gpu %s -o - | FileCheck %s --check-prefix=ASM

; Documents the current static HLSL binding model for slots 0..3:
;   register(u0/t0/b0) -> r1
;   register(u1/t1/b1) -> r2
;   register(u2/t2/b2) -> r3
;   register(u3/t3/b3) -> r4
;
; Binding space is not modeled yet; only the binding slot is used.

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p2:32:32-p3:32:32-i32:32-f32:32-n32"

%__cblayout_Params = type <{ i32 }>

@u0.str = private unnamed_addr constant [3 x i8] c"u0\00"
@t1.str = private unnamed_addr constant [3 x i8] c"t1\00"
@b2.str = private unnamed_addr constant [3 x i8] c"b2\00"
@u3.str = private unnamed_addr constant [3 x i8] c"u3\00"
@space.str = private unnamed_addr constant [6 x i8] c"space\00"

@Params.cb = local_unnamed_addr global target("dx.CBuffer", %__cblayout_Params) poison
@value = external hidden local_unnamed_addr addrspace(2) global i32, align 4

declare i32 @llvm.dx.thread.id(i32)
declare target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32, i32, i32, i32, ptr)
declare target("dx.TypedBuffer", i32, 0, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_0_0_0t(i32, i32, i32, i32, ptr)
declare target("dx.CBuffer", %__cblayout_Params) @llvm.dx.resource.handlefrombinding.tdx.CBuffer_s___cblayout_Paramsst(i32, i32, i32, i32, ptr)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0), i32)
declare { i32, i1 } @llvm.dx.resource.load.typedbuffer.i32.tdx.TypedBuffer_i32_0_0_0t(target("dx.TypedBuffer", i32, 0, 0, 0), i32)

define void @direct_slots() #0 {
; IR-LABEL: define void @direct_slots
; IR: call i32 @llvm.read_register.i32(metadata [[R1:![0-9]+]])
; IR: call i32 @llvm.read_register.i32(metadata [[R2:![0-9]+]])
; IR: call i32 @llvm.read_register.i32(metadata [[R3:![0-9]+]])
; IR: call i32 @llvm.read_register.i32(metadata [[R4:![0-9]+]])
;
; ASM-LABEL: direct_slots:
; ASM: add {{r[0-9]+}}, r1, {{r[0-9]+}}
; ASM: add {{r[0-9]+}}, r2, {{r[0-9]+}}
; ASM: ld_scatter {{r[0-9]+}}, [r3 + 0x0]
; ASM: add {{r[0-9]+}}, r4, {{r[0-9]+}}
entry:
  %id = call i32 @llvm.dx.thread.id(i32 0)
  %h0 = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @u0.str)
  %h1 = call target("dx.TypedBuffer", i32, 0, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_0_0_0t(i32 0, i32 1, i32 1, i32 0, ptr @t1.str)
  %h2 = call target("dx.CBuffer", %__cblayout_Params) @llvm.dx.resource.handlefrombinding.tdx.CBuffer_s___cblayout_Paramsst(i32 0, i32 2, i32 1, i32 0, ptr @b2.str)
  store target("dx.CBuffer", %__cblayout_Params) %h2, ptr @Params.cb, align 4
  %h3 = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 3, i32 1, i32 0, ptr @u3.str)

  %p0 = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %h0, i32 %id)
  store i32 %id, ptr %p0, align 4

  %l1 = call { i32, i1 } @llvm.dx.resource.load.typedbuffer.i32.tdx.TypedBuffer_i32_0_0_0t(target("dx.TypedBuffer", i32, 0, 0, 0) %h1, i32 %id)
  %v1 = extractvalue { i32, i1 } %l1, 0
  %v2 = load i32, ptr addrspace(2) @value, align 4
  %sum = add i32 %v1, %v2

  %p3 = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %h3, i32 %id)
  store i32 %sum, ptr %p3, align 4
  ret void
}

define void @nonzero_space_uses_bind_slot() #0 {
; IR-LABEL: define void @nonzero_space_uses_bind_slot
; IR: call i32 @llvm.read_register.i32(metadata [[R3]])
;
; ASM-LABEL: nonzero_space_uses_bind_slot:
; ASM: add {{r[0-9]+}}, r3, {{r[0-9]+}}
entry:
  %id = call i32 @llvm.dx.thread.id(i32 0)
  %h = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 9, i32 2, i32 1, i32 0, ptr @space.str)
  %p = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %h, i32 %id)
  store i32 %id, ptr %p, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }

!hlsl.cbs = !{!0}
!0 = !{ptr @Params.cb, ptr addrspace(2) @value}

; IR-DAG: [[R1]] = !{!"r1"}
; IR-DAG: [[R2]] = !{!"r2"}
; IR-DAG: [[R3]] = !{!"r3"}
; IR-DAG: [[R4]] = !{!"r4"}
