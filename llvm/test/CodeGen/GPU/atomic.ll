; RUN: llc -march=gpu < %s | FileCheck %s

define i32 @test_atomic_add(ptr %p, i32 %val) {
; CHECK: atomic
  %r = atomicrmw add ptr %p, i32 %val seq_cst
  ret i32 %r
}

define i32 @test_atomic_xor(ptr %p, i32 %val) {
; CHECK: atomic
  %r = atomicrmw xor ptr %p, i32 %val seq_cst
  ret i32 %r
}

define i32 @test_atomic_or(ptr %p, i32 %val) {
; CHECK: atomic
  %r = atomicrmw or ptr %p, i32 %val seq_cst
  ret i32 %r
}

define i32 @test_atomic_and(ptr %p, i32 %val) {
; CHECK: atomic
  %r = atomicrmw and ptr %p, i32 %val seq_cst
  ret i32 %r
}

define i32 @test_atomic_xchg(ptr %p, i32 %val) {
; CHECK: atomic
  %r = atomicrmw xchg ptr %p, i32 %val seq_cst
  ret i32 %r
}

define i32 @test_atomic_max(ptr %p, i32 %val) {
; CHECK: atomic
  %r = atomicrmw max ptr %p, i32 %val seq_cst
  ret i32 %r
}

define i32 @test_atomic_umin(ptr %p, i32 %val) {
; CHECK: atomic
  %r = atomicrmw umin ptr %p, i32 %val seq_cst
  ret i32 %r
}

define i32 @test_cmpxchg(ptr %p, i32 %cmp, i32 %new) {
; CHECK: atomic_cas
  %pair = cmpxchg ptr %p, i32 %cmp, i32 %new seq_cst seq_cst
  %r = extractvalue { i32, i1 } %pair, 0
  ret i32 %r
}
