; RUN: llc -march=gpu %s -o - | FileCheck %s

target triple = "spir"

declare spir_func void @llvm.spv.group.memory.barrier()
declare spir_func void @llvm.spv.device.memory.barrier()
declare spir_func void @llvm.spv.all.memory.barrier()
declare spir_func void @llvm.spv.group.memory.barrier.with.group.sync()
declare spir_func void @llvm.spv.device.memory.barrier.with.group.sync()
declare spir_func void @llvm.spv.all.memory.barrier.with.group.sync()
declare token @llvm.experimental.convergence.entry()

define spir_kernel void @barrier_variants() #0 {
entry:
  %tok = call token @llvm.experimental.convergence.entry()
  call spir_func void @llvm.spv.group.memory.barrier() [ "convergencectrl"(token %tok) ]
  call spir_func void @llvm.spv.device.memory.barrier() [ "convergencectrl"(token %tok) ]
  call spir_func void @llvm.spv.all.memory.barrier() [ "convergencectrl"(token %tok) ]
  call spir_func void @llvm.spv.group.memory.barrier.with.group.sync() [ "convergencectrl"(token %tok) ]
  call spir_func void @llvm.spv.device.memory.barrier.with.group.sync() [ "convergencectrl"(token %tok) ]
  call spir_func void @llvm.spv.all.memory.barrier.with.group.sync() [ "convergencectrl"(token %tok) ]
  ret void
}

; CHECK-LABEL: barrier_variants:
; CHECK: mem_fence	1
; CHECK: mem_fence	2
; CHECK: mem_fence	3
; CHECK: barrier	1
; CHECK: barrier	2
; CHECK: barrier	3
; CHECK: halt

attributes #0 = { convergent }
