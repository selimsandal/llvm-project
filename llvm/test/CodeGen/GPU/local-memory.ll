; RUN: llc -march=gpu < %s | FileCheck %s

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

define void @kernel(ptr %out, ptr addrspace(3) %local) {
; CHECK-LABEL: kernel:
; CHECK: ld_local {{r[0-9]+}}, [r2 + 0x0]
; CHECK: add
; CHECK: st_local [r2 + 0x0], {{r[0-9]+}}
; CHECK: st_scatter [r1 + 0x0], {{r[0-9]+}}
; CHECK: halt
entry:
  %v = load i32, ptr addrspace(3) %local
  %v1 = add i32 %v, 1
  store i32 %v1, ptr addrspace(3) %local
  store i32 %v1, ptr %out
  ret void
}
