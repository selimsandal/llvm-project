; RUN: llc -march=gpu < %s | FileCheck %s
; Test that IF/ELSE correctly lowers select and comparisons

; Integer select: icmp + select lowers to CMP + SEL
define void @test_select_ne(i32 %cond, ptr %out) {
; CHECK-LABEL: test_select_ne:
; CHECK: cmp
; CHECK: sel
; CHECK: st_scatter
  %c = icmp ne i32 %cond, 0
  %v = select i1 %c, i32 42, i32 99
  store i32 %v, ptr %out
  ret void
}

; Float select via fmin pattern
define void @test_fmin_select(float %a, float %b, ptr %out) {
; CHECK-LABEL: test_fmin_select:
; CHECK: fmin
  %c = fcmp olt float %a, %b
  %v = select i1 %c, float %a, float %b
  store float %v, ptr %out
  ret void
}

; Diamond if/else: both paths store different values
define void @test_diamond(i32 %x, ptr %out) {
; CHECK-LABEL: test_diamond:
; CHECK: st_scatter
  %c = icmp sgt i32 %x, 100
  br i1 %c, label %then, label %else

then:
  store i32 1, ptr %out
  br label %merge

else:
  store i32 2, ptr %out
  br label %merge

merge:
  ret void
}
