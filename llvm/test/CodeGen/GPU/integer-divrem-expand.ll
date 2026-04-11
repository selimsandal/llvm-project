; RUN: llc -march=gpu < %s | FileCheck %s

; Regression for: variable-divisor `sdiv` / `udiv` / `srem` / `urem`
; used to crash llc with `error: no libcall available for sdiv` because
; the backend marked them Expand at the DAG level but generic Legalize
; only knows how to expand div/rem to a libcall (which the GPU runtime
; does not provide). The fix sets `setMaxDivRemBitWidthSupported(0)` so
; the IR-level `ExpandIRInsts` pass expands all integer div/rem before
; ISel into the bit-by-bit shift/subtract sequence from
; IntegerDivision.cpp. This is what unblocks Rodinia mergesort's
; `gid / threadsPerDiv` (where the divisor is a runtime value).

target triple = "spir"

define dso_local spir_kernel void @sdiv_test(ptr addrspace(1) %a,
                                             ptr addrspace(1) %b,
                                             ptr addrspace(1) %out) {
; CHECK-LABEL: sdiv_test:
; CHECK:       halt
; CHECK-NOT:   __divsi3
; CHECK-NOT:   __aeabi_idiv
  %x = load i32, ptr addrspace(1) %a, align 4
  %y = load i32, ptr addrspace(1) %b, align 4
  %q = sdiv i32 %x, %y
  store i32 %q, ptr addrspace(1) %out, align 4
  ret void
}

define dso_local spir_kernel void @udiv_test(ptr addrspace(1) %a,
                                             ptr addrspace(1) %b,
                                             ptr addrspace(1) %out) {
; CHECK-LABEL: udiv_test:
; CHECK:       halt
; CHECK-NOT:   __udivsi3
  %x = load i32, ptr addrspace(1) %a, align 4
  %y = load i32, ptr addrspace(1) %b, align 4
  %q = udiv i32 %x, %y
  store i32 %q, ptr addrspace(1) %out, align 4
  ret void
}

define dso_local spir_kernel void @srem_test(ptr addrspace(1) %a,
                                             ptr addrspace(1) %b,
                                             ptr addrspace(1) %out) {
; CHECK-LABEL: srem_test:
; CHECK:       halt
; CHECK-NOT:   __modsi3
  %x = load i32, ptr addrspace(1) %a, align 4
  %y = load i32, ptr addrspace(1) %b, align 4
  %r = srem i32 %x, %y
  store i32 %r, ptr addrspace(1) %out, align 4
  ret void
}
