; RUN: llc -march=gpu < %s | FileCheck %s

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare void @llvm.gpu.workgroup.sync(i32)
declare void @llvm.gpu.mem.fence(i32)

define void @kernel(ptr %out) {
; CHECK-LABEL: kernel:
; CHECK: barrier 1
; CHECK: mem_fence 2
; CHECK: st_scatter [r1 + 0x0]
; CHECK: halt
entry:
  call void @llvm.gpu.workgroup.sync(i32 1)
  call void @llvm.gpu.mem.fence(i32 2)
  store i32 42, ptr %out
  ret void
}
