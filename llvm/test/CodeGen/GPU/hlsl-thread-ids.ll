; RUN: llc -march=gpu %s -o - | FileCheck %s

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare i32 @llvm.dx.thread.id(i32)
declare i32 @llvm.dx.group.id(i32)
declare i32 @llvm.dx.thread.id.in.group(i32)
declare i32 @llvm.dx.flattened.thread.id.in.group()
declare i32 @llvm.dx.wave.getlaneindex()

define void @main(ptr %out) #0 {
; CHECK-LABEL: main:
; CHECK-DAG: getsr {{r[0-9]+}}, 50
; CHECK-DAG: getsr {{r[0-9]+}}, 53
; CHECK-DAG: getsr {{r[0-9]+}}, 62
; CHECK-DAG: getsr {{r[0-9]+}}, 48
; CHECK-DAG: getsr {{r[0-9]+}}, 61
; CHECK-DAG: getsr {{r[0-9]+}}, 60
; CHECK-DAG: getsr {{r[0-9]+}}, 63
; CHECK-DAG: getsr {{r[0-9]+}}, 64
; CHECK-DAG: getsr {{r[0-9]+}}, 51
; CHECK-DAG: getsr {{r[0-9]+}}, 52
; CHECK-DAG: getsr {{r[0-9]+}}, 59
; CHECK-DAG: mul
; CHECK: sub
; CHECK: st_scatter
; CHECK: halt
entry:
  %dispatch_z = call i32 @llvm.dx.thread.id(i32 2)
  %group_x = call i32 @llvm.dx.group.id(i32 0)
  %local_y = call i32 @llvm.dx.thread.id.in.group(i32 1)
  %flat = call i32 @llvm.dx.flattened.thread.id.in.group()
  %lane = call i32 @llvm.dx.wave.getlaneindex()
  %s0 = add i32 %dispatch_z, %group_x
  %s1 = add i32 %s0, %local_y
  %s2 = add i32 %s1, %flat
  %s3 = add i32 %s2, %lane
  store i32 %s3, ptr %out
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,4,2" }
