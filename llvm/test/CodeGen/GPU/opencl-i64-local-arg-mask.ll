; RUN: llc -march=gpu < %s | FileCheck %s

target triple = "spir"

define spir_kernel void @i64_local_mask(i64 %x, ptr addrspace(3) %scratch, i32 %a, i32 %b)
    !kernel_arg_addr_space !0 {
; CHECK-LABEL: __gpu_kernel_metadata:
; CHECK-NEXT: .long 1297436743
; CHECK-NEXT: .short 1
; CHECK-NEXT: .short 24
; CHECK-NEXT: .long 324
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 5
; CHECK-NEXT: .long 4
entry:
  ret void
}

!0 = !{i32 0, i32 3, i32 0, i32 0}
