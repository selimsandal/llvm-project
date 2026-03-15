; RUN: llc -march=gpu < %s | FileCheck %s
; Test UITOF and FTOU instruction selection

define void @test_uitof(i32 %x, ptr %out) {
; CHECK-LABEL: test_uitof:
; CHECK: uitof
  %f = uitofp i32 %x to float
  %bits = bitcast float %f to i32
  store i32 %bits, ptr %out
  ret void
}

define void @test_ftou(ptr %fptr, ptr %out) {
; CHECK-LABEL: test_ftou:
; CHECK: ftou
  %x = load float, ptr %fptr
  %u = fptoui float %x to i32
  store i32 %u, ptr %out
  ret void
}

; Round-trip: uitof then ftou should preserve values in [0, 2^24]
define void @test_roundtrip(i32 %x, ptr %out) {
; CHECK-LABEL: test_roundtrip:
; CHECK: uitof
; CHECK: ftou
  %f = uitofp i32 %x to float
  %u = fptoui float %f to i32
  store i32 %u, ptr %out
  ret void
}
