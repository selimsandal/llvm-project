; RUN: llc -march=gpu < %s | FileCheck %s

; Simple i32 load
define void @test_load_i32(ptr %p, ptr %out) {
; CHECK: ld_scatter
  %v = load i32, ptr %p
  store i32 %v, ptr %out
  ret void
}

; Simple i32 store
define void @test_store_i32(ptr %p, i32 %val) {
; CHECK: st_scatter
  store i32 %val, ptr %p
  ret void
}

; Load with offset
define void @test_load_offset(ptr %p, ptr %out) {
; CHECK: ld_scatter{{.*}}0xc
  %gep = getelementptr i32, ptr %p, i32 3
  %v = load i32, ptr %gep
  store i32 %v, ptr %out
  ret void
}

; Store with offset
define void @test_store_offset(ptr %p, i32 %val) {
; CHECK: st_scatter{{.*}}0x10
  %gep = getelementptr i32, ptr %p, i32 4
  store i32 %val, ptr %gep
  ret void
}

; f32 load
define void @test_load_f32(ptr %p, ptr %out) {
; CHECK: ld_scatter
  %v = load float, ptr %p
  store float %v, ptr %out
  ret void
}

; f32 store
define void @test_store_f32(ptr %p, float %val) {
; CHECK: st_scatter
  store float %val, ptr %p
  ret void
}

; f32 load with offset
define void @test_load_f32_offset(ptr %p, ptr %out) {
; CHECK: ld_scatter{{.*}}0x8
  %gep = getelementptr float, ptr %p, i32 2
  %v = load float, ptr %gep
  store float %v, ptr %out
  ret void
}

; f32 store with offset
define void @test_store_f32_offset(ptr %p, float %val) {
; CHECK: st_scatter{{.*}}0xc
  %gep = getelementptr float, ptr %p, i32 3
  store float %val, ptr %gep
  ret void
}
