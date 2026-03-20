; RUN: llc -march=gpu < %s | FileCheck %s

define i32 @signed_lt(i32 %a, i32 %b) {
; CHECK-LABEL: signed_lt:
; CHECK: movi
; CHECK: cmp
; CHECK: sel
; CHECK: smin
; CHECK: movi
; CHECK: cmp
; CHECK: sel
; CHECK: and
entry:
  %cmp = icmp slt i32 %a, %b
  %z = zext i1 %cmp to i32
  ret i32 %z
}

define i32 @signed_le(i32 %a, i32 %b) {
; CHECK-LABEL: signed_le:
; CHECK: smin
; CHECK: cmp
; CHECK: sel
entry:
  %cmp = icmp sle i32 %a, %b
  %z = zext i1 %cmp to i32
  ret i32 %z
}

define i32 @signed_gt(i32 %a, i32 %b) {
; CHECK-LABEL: signed_gt:
; CHECK: movi
; CHECK: cmp
; CHECK: sel
; CHECK: smax
; CHECK: movi
; CHECK: cmp
; CHECK: sel
; CHECK: and
entry:
  %cmp = icmp sgt i32 %a, %b
  %z = zext i1 %cmp to i32
  ret i32 %z
}

define void @signed_lt_branch(i32 %a, i32 %b, ptr %out) {
; CHECK-LABEL: signed_lt_branch:
; CHECK: smax
; CHECK: movi
; CHECK: cmp
; CHECK: sel
; CHECK: cmp
; CHECK: goto
entry:
  %cmp = icmp slt i32 %a, %b
  br i1 %cmp, label %then, label %merge

then:
  store i32 1, ptr %out
  br label %merge

merge:
  ret void
}
