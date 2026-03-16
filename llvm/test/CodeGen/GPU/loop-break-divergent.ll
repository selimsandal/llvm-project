; RUN: llc -march=gpu < %s | FileCheck %s

; Loop with break and divergent exit phi values.
; Exit-block instructions are cloned before BREAK so breaking lanes have
; correct register values. The exhausted-path PHI copy (SELi) runs inside
; the loop before ENDLOOP with the correct exec_mask. BREAK sets break_mask
; and clears exec_mask (no jump) — ENDIF/ELSE apply ~break_mask to prevent
; re-activating broken lanes.

define void @loop_break_phi(i32 %n, ptr %arr, ptr %out) {
; Exit-block store is cloned before BREAK
; CHECK: loop
; CHECK: st_scatter
; CHECK-NEXT: break
; Exhausted-path PHI copy (SEL) runs inside the loop before ENDLOOP
; CHECK: sel
; CHECK-NEXT: endloop
; No stale PHI copies after ENDLOOP (would overwrite BREAKed lanes)
; CHECK-NOT: sel
; CHECK-NOT: movi
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
