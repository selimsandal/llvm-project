; RUN: llc -march=gpu -mtriple=gpu-none-none -filetype=obj %s -o %t.o
; RUN: llc -march=gpu -mtriple=gpu-none-none %s -o - | FileCheck %s

target triple = "spirv-unknown-vulkan-compute"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare i32 @llvm.spv.thread.id.i32(i32)
declare i32 @llvm.spv.group.id.i32(i32)
declare i32 @llvm.spv.thread.id.in.group.i32(i32)
declare i32 @llvm.spv.flattened.thread.id.in.group()

define void @main(ptr %out) #0 {
; CHECK-LABEL: main:
; CHECK-DAG: getsr {{r[0-9]+}}, 50
; CHECK-DAG: getsr {{r[0-9]+}}, 53
; CHECK-DAG: getsr {{r[0-9]+}}, 62
; CHECK-DAG: getsr {{r[0-9]+}}, 48
; CHECK-DAG: getsr {{r[0-9]+}}, 61
; CHECK-DAG: getsr {{r[0-9]+}}, 51
; CHECK-DAG: getsr {{r[0-9]+}}, 52
; CHECK-DAG: getsr {{r[0-9]+}}, 59
; CHECK-DAG: mul
; CHECK: st_scatter
; CHECK: halt
entry:
  %dispatch_z = call i32 @llvm.spv.thread.id.i32(i32 2)
  %group_x = call i32 @llvm.spv.group.id.i32(i32 0)
  %local_y = call i32 @llvm.spv.thread.id.in.group.i32(i32 1)
  %flat = call i32 @llvm.spv.flattened.thread.id.in.group()
  %s0 = add i32 %dispatch_z, %group_x
  %s1 = add i32 %s0, %local_y
  %s2 = add i32 %s1, %flat
  store i32 %s2, ptr %out
  ret void
}

attributes #0 = { "hlsl.numthreads"="8,4,2" "hlsl.shader"="compute" }
