; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s

define void @test_memory_barrier_variants() {
entry:
  ; CHECK: call void @dx.op.barrier(i32 80, i32 8)
  call void @llvm.dx.group.memory.barrier()
  ; CHECK: call void @dx.op.barrier(i32 80, i32 2)
  call void @llvm.dx.device.memory.barrier()
  ; CHECK: call void @dx.op.barrier(i32 80, i32 10)
  call void @llvm.dx.all.memory.barrier()
  ; CHECK: call void @dx.op.barrier(i32 80, i32 9)
  call void @llvm.dx.group.memory.barrier.with.group.sync()
  ; CHECK: call void @dx.op.barrier(i32 80, i32 3)
  call void @llvm.dx.device.memory.barrier.with.group.sync()
  ; CHECK: call void @dx.op.barrier(i32 80, i32 11)
  call void @llvm.dx.all.memory.barrier.with.group.sync()
  ret void
}
