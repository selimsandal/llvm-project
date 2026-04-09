; RUN: llc -march=gpu < %s | FileCheck %s

target triple = "spir"

define spir_kernel void @indirect_i64(ptr addrspace(1) %out, i64 %x, i32 %a, i32 %b)
    !kernel_arg_addr_space !0 {
; CHECK-LABEL: indirect_i64:
; CHECK: ld_scatter
; CHECK: [r1 + 0x10]
; CHECK-LABEL: __gpu_kernel_metadata:
; CHECK-NEXT: .long 1297436743
; CHECK-NEXT: .short 1
; CHECK-NEXT: .short 24
; CHECK-NEXT: .long 320
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 5
; CHECK-NEXT: .long 0
entry:
  %x32 = trunc i64 %x to i32
  %sum0 = add i32 %x32, %a
  %sum1 = add i32 %sum0, %b
  store i32 %sum1, ptr addrspace(1) %out, align 4
  ret void
}

!0 = !{i32 1, i32 0, i32 0, i32 0}
