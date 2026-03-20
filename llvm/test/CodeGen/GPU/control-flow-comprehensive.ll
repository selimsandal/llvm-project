; RUN: llc -march=gpu < %s | FileCheck %s
; Comprehensive control flow tests for GPU structured control flow.

; Test 1: Simple loop with counter
define void @test_loop_sum(ptr %out) {
; CHECK-LABEL: test_loop_sum:
; CHECK: while
; CHECK: break
; CHECK: jump
; CHECK: join
; CHECK: st_scatter
entry:
  br label %loop
loop:
  %sum = phi i32 [ 0, %entry ], [ %new_sum, %loop ]
  %i = phi i32 [ 1, %entry ], [ %next, %loop ]
  %new_sum = add i32 %sum, %i
  %next = add i32 %i, 1
  %done = icmp ult i32 %next, 11
  br i1 %done, label %loop, label %exit
exit:
  store i32 %new_sum, ptr %out
  ret void
}

; Test 2: Loop with simple break (single exit value)
define void @test_loop_break(ptr %out) {
; CHECK-LABEL: test_loop_break:
; CHECK: while
; CHECK: break
; CHECK: jump
; CHECK: join
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %cont ]
  %cmp = icmp eq i32 %i, 3
  br i1 %cmp, label %exit, label %cont
cont:
  %next = add i32 %i, 1
  %cmp2 = icmp ult i32 %next, 10
  br i1 %cmp2, label %loop, label %exit
exit:
  %result = phi i32 [ 42, %loop ], [ 99, %cont ]
  store i32 %result, ptr %out
  ret void
}

; Test 3: Diamond if/else — integer
define void @test_diamond_int(i32 %x, ptr %out) {
; CHECK-LABEL: test_diamond_int:
; CHECK: cmp
; CHECK: st_scatter
entry:
  %cmp = icmp sgt i32 %x, 100
  br i1 %cmp, label %then, label %else
then:
  br label %merge
else:
  br label %merge
merge:
  %val = phi i32 [ 170, %then ], [ 187, %else ]
  store i32 %val, ptr %out
  ret void
}

; Test 4: Diamond if/else — float comparison
define void @test_diamond_float(float %x, ptr %out) {
; CHECK-LABEL: test_diamond_float:
; CHECK: cmp
; CHECK: st_scatter
entry:
  %cmp = fcmp ogt float %x, 0.0
  br i1 %cmp, label %then, label %else
then:
  br label %merge
else:
  br label %merge
merge:
  %val = phi i32 [ 170, %then ], [ 187, %else ]
  store i32 %val, ptr %out
  ret void
}

; Test 5: Triangle (then-only, no else body)
define void @test_triangle(i32 %x, ptr %out) {
; CHECK-LABEL: test_triangle:
; CHECK: cmp
; CHECK: sel
; CHECK: st_scatter
entry:
  store i32 99, ptr %out
  %cmp = icmp sgt i32 %x, 50
  br i1 %cmp, label %then, label %merge
then:
  store i32 42, ptr %out
  br label %merge
merge:
  ret void
}

declare i32 @llvm.read_register.i32(metadata)
!0 = !{!"r0"}
