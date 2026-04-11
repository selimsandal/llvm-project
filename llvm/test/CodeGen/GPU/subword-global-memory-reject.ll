; RUN: not llc -march=gpu < %s 2>&1 | FileCheck %s

; The sub-word global-memory lowering pass may rewrite ordinary i8/i16
; stores into a word-sized load-mask-store, but that sequence is not a
; valid implementation for atomic or volatile stores. Reject those cases
; explicitly instead of silently miscompiling them.

target triple = "spir"

define dso_local spir_kernel void @volatile_i8_store(ptr addrspace(1) %p, i8 %v) {
; CHECK: error: GPU backend does not support atomic or volatile sub-word global stores
  store volatile i8 %v, ptr addrspace(1) %p, align 1
  ret void
}
