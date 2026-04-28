; RUN: llc -march=gpu %s -o - | FileCheck %s

target triple = "dxilv1.0-pc-shadermodel6.0-compute"

@s = external hidden addrspace(3) global [8 x i32], align 4

declare i32 @llvm.dx.thread.id.in.group(i32)
declare void @llvm.dx.group.memory.barrier.with.group.sync()

define void @CSMain() #0 {
entry:
  %tid = call i32 @llvm.dx.thread.id.in.group(i32 0)
  %store.ptr = getelementptr inbounds [8 x i32], ptr addrspace(3) @s, i32 0, i32 %tid
  store i32 %tid, ptr addrspace(3) %store.ptr, align 4
  call void @llvm.dx.group.memory.barrier.with.group.sync()
  %next = add i32 %tid, 1
  %next.masked = and i32 %next, 7
  %load.ptr = getelementptr inbounds [8 x i32], ptr addrspace(3) @s, i32 0, i32 %next.masked
  %val = load i32, ptr addrspace(3) %load.ptr, align 4
  store i32 %val, ptr addrspace(3) %store.ptr, align 4
  ret void
}

attributes #0 = { convergent noinline norecurse "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }

; CHECK-LABEL: CSMain:
; CHECK: getsr
; CHECK: st_local
; CHECK: barrier	1
; CHECK: ld_local
; CHECK: st_local
; CHECK: halt
