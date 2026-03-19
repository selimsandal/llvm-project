; RUN: llc -march=gpu < %s | FileCheck %s

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare i32 @_Z13get_global_idj(i32)
declare i32 @_Z12get_group_idj(i32)
declare i32 @_Z12get_local_idj(i32)
declare i32 @_Z15get_global_sizej(i32)
declare i32 @_Z14get_num_groupsj(i32)

define void @kernel(ptr %out) {
; CHECK-LABEL: kernel:
; CHECK-DAG: getsr {{r[0-9]+}}, 49
; CHECK-DAG: getsr {{r[0-9]+}}, 52
; CHECK-DAG: getsr {{r[0-9]+}}, 61
; CHECK-DAG: getsr {{r[0-9]+}}, 48
; CHECK-DAG: getsr {{r[0-9]+}}, 55
; CHECK-DAG: getsr {{r[0-9]+}}, 56
; CHECK-DAG: mul
; CHECK-DAG: add
; CHECK: st_scatter
; CHECK: halt
entry:
  %gid_y = call i32 @_Z13get_global_idj(i32 1)
  %lid_y = call i32 @_Z12get_local_idj(i32 1)
  %grp_x = call i32 @_Z12get_group_idj(i32 0)
  %gsize_y = call i32 @_Z15get_global_sizej(i32 1)
  %ng_z = call i32 @_Z14get_num_groupsj(i32 2)
  %s0 = add i32 %gid_y, %lid_y
  %s1 = add i32 %s0, %grp_x
  %s2 = add i32 %s1, %gsize_y
  %s3 = add i32 %s2, %ng_z
  store i32 %s3, ptr %out
  ret void
}
