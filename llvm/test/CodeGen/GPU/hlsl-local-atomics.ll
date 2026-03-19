; RUN: llc -march=gpu %s -o - | FileCheck %s

target triple = "dxilv1.0-pc-shadermodel6.0-compute"

@s = external hidden addrspace(3) global [8 x i32], align 4
@u = external hidden addrspace(3) global [8 x i32], align 4

declare i32 @llvm.dx.thread.id.in.group(i32)
declare void @InterlockedAdd(ptr addrspace(3), i32, ptr)
declare void @InterlockedMax_unsigned(ptr addrspace(3), i32, ptr)
declare void @InterlockedCompareExchange(ptr addrspace(3), i32, i32, ptr)

define void @CSMain() #0 {
entry:
  %tid = call i32 @llvm.dx.thread.id.in.group(i32 0)

  %s.ptr = getelementptr inbounds [8 x i32], ptr addrspace(3) @s, i32 0, i32 %tid
  %u.ptr = getelementptr inbounds [8 x i32], ptr addrspace(3) @u, i32 0, i32 %tid
  %old.add = alloca i32, align 4
  %old.max = alloca i32, align 4
  %old.cas = alloca i32, align 4

  call void @InterlockedAdd(ptr addrspace(3) %s.ptr, i32 1, ptr %old.add)
  call void @InterlockedMax_unsigned(ptr addrspace(3) %u.ptr, i32 %tid, ptr %old.max)
  call void @InterlockedCompareExchange(ptr addrspace(3) %s.ptr, i32 0, i32 42, ptr %old.cas)
  ret void
}

attributes #0 = { convergent noinline norecurse "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }

; CHECK-LABEL: CSMain:
; CHECK: atomic_local
; CHECK: atomic_local
; CHECK: atomic_local_cas
; CHECK: halt
