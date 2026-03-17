; RUN: llc -march=gpu < %s | FileCheck %s

; Simple counted loop
define i32 @test_sum_loop(ptr %arr, i32 %n) {
; CHECK: while
; CHECK: break
; CHECK: jump
; CHECK: join
; CHECK: halt
entry:
  %cmp0 = icmp sgt i32 %n, 0
  br i1 %cmp0, label %loop, label %exit

loop:
  %i = phi i32 [0, %entry], [%i.next, %loop]
  %sum = phi i32 [0, %entry], [%sum.next, %loop]
  %gep = getelementptr i32, ptr %arr, i32 %i
  %v = load i32, ptr %gep
  %sum.next = add i32 %sum, %v
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  %result = phi i32 [0, %entry], [%sum.next, %loop]
  ret i32 %result
}
