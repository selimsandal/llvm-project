; RUN: llc -march=gpu < %s | FileCheck %s

; Loop with break and divergent exit phi values.
; Exit-block instructions are cloned before BREAK so breaking lanes have
; correct register values. The exhausted-path PHI copy runs inside the loop
; with the correct exec_mask, and a scratch MOVI may be materialized for the
; immediate select because control-flow lowering runs after register allocation.

define void @loop_break_phi(i32 %n, ptr %arr, ptr %out) {
; CHECK-LABEL: loop_break_phi:
; Exit-block work is still lowered inside the loop before the break/jump pair.
; CHECK: while
; Exhausted-path PHI copy runs inside the loop before the matching JOIN.
; CHECK: sel
; CHECK: break
; CHECK: jump
; CHECK: join
; CHECK: st_scatter
; CHECK: halt
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.next, %latch]
  %ptr = getelementptr i32, ptr %arr, i32 %i
  %val = load i32, ptr %ptr
  %hit = icmp eq i32 %val, 42
  br i1 %hit, label %found, label %latch

found:
  br label %done

latch:
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %not_found

not_found:
  br label %done

done:
  %result = phi i32 [%val, %found], [0, %not_found]
  store i32 %result, ptr %out
  ret void
}
