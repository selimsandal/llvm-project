; RUN: llc -march=gpu < %s | FileCheck %s

; Diamond if-then-else: should produce CMP + IF + ELSE + ENDIF
define void @test_diamond(i32 %a, i32 %b, ptr %out) {
; CHECK: cmp
; CHECK: goto
; CHECK: st_scatter
; CHECK: st_scatter
; CHECK: join
; CHECK: halt
entry:
  %cmp = icmp slt i32 %a, %b
  br i1 %cmp, label %then, label %else

then:
  store i32 %a, ptr %out
  br label %merge

else:
  store i32 %b, ptr %out
  br label %merge

merge:
  ret void
}

; Triangle if-then (no else): should produce CMP + IF + ENDIF
define void @test_triangle(i32 %a, i32 %b, ptr %out) {
; CHECK: cmp
; CHECK: goto
; CHECK: st_scatter
; CHECK: join
; CHECK: halt
entry:
  %cmp = icmp slt i32 %a, %b
  br i1 %cmp, label %then, label %merge

then:
  store i32 %a, ptr %out
  br label %merge

merge:
  ret void
}

; Nested if-then-else
define void @test_nested(i32 %a, i32 %b, i32 %c, ptr %out) {
; CHECK: cmp
; CHECK: goto
; CHECK: cmp
; CHECK: goto
; CHECK: st_scatter
; CHECK: join
; CHECK: halt
entry:
  %cmp1 = icmp slt i32 %a, %b
  br i1 %cmp1, label %outer_then, label %outer_else

outer_then:
  %cmp2 = icmp slt i32 %b, %c
  br i1 %cmp2, label %inner_then, label %inner_merge

inner_then:
  store i32 %c, ptr %out
  br label %inner_merge

inner_merge:
  br label %merge

outer_else:
  store i32 %a, ptr %out
  br label %merge

merge:
  ret void
}
