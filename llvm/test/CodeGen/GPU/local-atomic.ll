; RUN: llc -march=gpu < %s | FileCheck %s

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

define void @local_atomic_kernel(ptr addrspace(3) %p, ptr %out, i32 %v, i32 %cmp, i32 %swap) {
; CHECK-LABEL: local_atomic_kernel:
; CHECK: atomic_local {{r[0-9]+}}, [{{r[0-9]+}} + 0x0], {{r[0-9]+}}
; CHECK: st_scatter [{{r[0-9]+}} + 0x0], {{r[0-9]+}}
; CHECK: add {{r[0-9]+}}, {{r[0-9]+}}, 0x4
; CHECK: atomic_local_cas {{r[0-9]+}}, [{{r[0-9]+}} + 0x0], {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: st_scatter [{{r[0-9]+}} + 0x4], {{r[0-9]+}}
; CHECK: halt
entry:
  %old.add = atomicrmw add ptr addrspace(3) %p, i32 %v seq_cst
  store i32 %old.add, ptr %out, align 4

  %p.next = getelementptr inbounds i32, ptr addrspace(3) %p, i32 1
  %pair = cmpxchg ptr addrspace(3) %p.next, i32 %cmp, i32 %swap seq_cst seq_cst
  %old.cas = extractvalue { i32, i1 } %pair, 0
  %out.next = getelementptr inbounds i32, ptr %out, i32 1
  store i32 %old.cas, ptr %out.next, align 4
  ret void
}
