; RUN: llc -march=gpu -filetype=obj %s -o %t.o
; RUN: llc -march=gpu %s -o - | FileCheck %s

; Test: HLSL compute shader lowering (vec_add)
;
; Equivalent HLSL:
;   RWStructuredBuffer<int> a : register(u0);  // → r1
;   RWStructuredBuffer<int> b : register(u1);  // → r2
;   RWStructuredBuffer<int> c : register(u2);  // → r3
;
;   [numthreads(8, 1, 1)]
;   void CSMain(uint id : SV_DispatchThreadID) {
;       c[id] = a[id] + b[id];
;   }

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

; Resource handle declarations (target extension types)
declare target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_0_0t(i32, i32, i32, i32, ptr)
declare target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32, i32, i32, i32, ptr)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_0_0t(target("dx.RawBuffer", i32, 0, 0), i32)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0), i32)

; Thread ID
declare i32 @llvm.dx.thread.id(i32)

; Buffer name strings (used by handlefrombinding)
@a.str = private unnamed_addr constant [2 x i8] c"a\00"
@b.str = private unnamed_addr constant [2 x i8] c"b\00"
@c.str = private unnamed_addr constant [2 x i8] c"c\00"

define void @main() #0 {
entry:
  ; Get thread ID (SV_DispatchThreadID.x)
  %id = call i32 @llvm.dx.thread.id(i32 0)

  ; Create resource handles
  ;   a = register(u0) → bind slot 0 → r1
  %handle_a = call target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_0_0t(i32 0, i32 0, i32 1, i32 0, ptr @a.str)
  ;   b = register(u1) → bind slot 1 → r2
  %handle_b = call target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_0_0t(i32 0, i32 1, i32 1, i32 0, ptr @b.str)
  ;   c = register(u2) → bind slot 2 → r3
  %handle_c = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 2, i32 1, i32 0, ptr @c.str)

  ; a[id]
  %ptr_a = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_0_0t(target("dx.RawBuffer", i32, 0, 0) %handle_a, i32 %id)
  %val_a = load i32, ptr %ptr_a

  ; b[id]
  %ptr_b = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_0_0t(target("dx.RawBuffer", i32, 0, 0) %handle_b, i32 %id)
  %val_b = load i32, ptr %ptr_b

  ; c[id] = a[id] + b[id]
  %sum = add i32 %val_a, %val_b
  %ptr_c = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %handle_c, i32 %id)
  store i32 %sum, ptr %ptr_c

  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,1,1" }

; CHECK: ld_scatter
; CHECK: ld_scatter
; CHECK: add
; CHECK: st_scatter
; CHECK: halt
