; RUN: llc -march=gpu %s -o - | FileCheck %s

target triple = "dxilv1.0-pc-shadermodel6.0-compute"

declare void @llvm.dx.group.memory.barrier()
declare void @llvm.dx.device.memory.barrier()
declare void @llvm.dx.all.memory.barrier()
declare void @llvm.dx.group.memory.barrier.with.group.sync()
declare void @llvm.dx.device.memory.barrier.with.group.sync()
declare void @llvm.dx.all.memory.barrier.with.group.sync()

define void @barrier_variants() #0 {
entry:
  call void @llvm.dx.group.memory.barrier()
  call void @llvm.dx.device.memory.barrier()
  call void @llvm.dx.all.memory.barrier()
  call void @llvm.dx.group.memory.barrier.with.group.sync()
  call void @llvm.dx.device.memory.barrier.with.group.sync()
  call void @llvm.dx.all.memory.barrier.with.group.sync()
  ret void
}

attributes #0 = { convergent noinline norecurse "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }

; CHECK-LABEL: barrier_variants:
; CHECK: mem_fence	1
; CHECK: mem_fence	2
; CHECK: mem_fence	3
; CHECK: barrier	1
; CHECK: barrier	2
; CHECK: barrier	3
; CHECK: halt
