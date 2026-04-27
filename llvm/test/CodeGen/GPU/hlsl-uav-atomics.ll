; RUN: llc -march=gpu %s -o - | FileCheck %s

target triple = "dxilv1.0-pc-shadermodel6.0-compute"

@counter.name = private unnamed_addr constant [8 x i8] c"Counter\00", align 1

declare target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32, i32, i32, i32, ptr)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0), i32)

declare void @InterlockedAdd(ptr, i32, ptr)
declare void @InterlockedExchange(ptr, i32, ptr)
declare void @InterlockedCompareExchange(ptr, i32, i32, ptr)

define void @uav_interlocked(ptr %counter, ptr %out, i32 %value) #0 {
entry:
  %old.add = alloca i32, align 4
  %old.xchg = alloca i32, align 4
  %old.cas = alloca i32, align 4

  call void @InterlockedAdd(ptr %counter, i32 1, ptr %old.add)
  call void @InterlockedExchange(ptr %counter, i32 %value, ptr %old.xchg)
  call void @InterlockedCompareExchange(ptr %counter, i32 0, i32 42, ptr %old.cas)

  %add = load i32, ptr %old.add, align 4
  %xchg = load i32, ptr %old.xchg, align 4
  %cas = load i32, ptr %old.cas, align 4
  %sum0 = add i32 %add, %xchg
  %sum1 = add i32 %sum0, %cas
  store i32 %sum1, ptr %out, align 4
  ret void
}

attributes #0 = { convergent noinline norecurse "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }

; CHECK-LABEL: uav_interlocked:
; CHECK-NOT: atomic_local
; CHECK: atomic
; CHECK-NOT: atomic_local
; CHECK: atomic
; CHECK-NOT: atomic_local
; CHECK: atomic_cas
; CHECK-NOT: atomic_local
; CHECK: halt

define void @uav_byteaddress_interlocked(i32 %byte.index) #0 {
entry:
  %handle = call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @counter.name)
  %counter = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %handle, i32 %byte.index)
  %old.add = atomicrmw add ptr %counter, i32 1 seq_cst, align 4
  %old.cas.pair = cmpxchg ptr %counter, i32 0, i32 42 seq_cst seq_cst, align 4
  %old.cas = extractvalue { i32, i1 } %old.cas.pair, 0
  %sum = add i32 %old.add, %old.cas
  store i32 %sum, ptr %counter, align 4
  ret void
}

; CHECK-LABEL: uav_byteaddress_interlocked:
; CHECK-NOT: atomic_local
; CHECK: atomic
; CHECK-NOT: atomic_local
; CHECK: atomic_cas
; CHECK-NOT: atomic_local
; CHECK: halt
