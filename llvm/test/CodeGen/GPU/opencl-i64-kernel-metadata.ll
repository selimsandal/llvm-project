; RUN: llc -march=gpu < %s | FileCheck %s

target triple = "spir"

define spir_kernel void @direct_i64(ptr addrspace(1) %out, i64 %x)
    !kernel_arg_addr_space !0 {
; CHECK-LABEL: direct_i64:
; CHECK: add
; CHECK: st_scatter
; CHECK: st_scatter
; CHECK-LABEL: __gpu_kernel_metadata:
; CHECK-NEXT: .long 1297436743
; CHECK-NEXT: .short 1
; CHECK-NEXT: .short 24
; CHECK-NEXT: .long 64
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 3
; CHECK-NEXT: .long 0
entry:
  %y = add i64 %x, 1
  store i64 %y, ptr addrspace(1) %out, align 8
  ret void
}

!0 = !{i32 1, i32 0}
