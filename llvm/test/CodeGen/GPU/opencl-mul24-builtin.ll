; RUN: llc -march=gpu < %s | FileCheck %s

; Regression for: OpenCL `mul24(a, b)` and `mad24(a, b, c)` used to
; reach llc as the unmangled libcalls `_Z5mul24ii` / `_Z5mad24iii`
; which the backend had no handler for, crashing call lowering with
; `std::length_error` out of LowerReturn (cold path). GPUSPIRVLowering
; now recognizes both names and rewrites them to plain i32 mul / mul+add.
; The 24-bit-only contract is satisfied by the GPU's full 32x32→32 mul.

target triple = "spir"

declare dso_local spir_func i32 @_Z5mul24ii(i32, i32) local_unnamed_addr
declare dso_local spir_func i32 @_Z5mad24iii(i32, i32, i32) local_unnamed_addr

define dso_local spir_kernel void @mul24_test(ptr addrspace(1) %a,
                                              ptr addrspace(1) %b,
                                              ptr addrspace(1) %out) {
; CHECK-LABEL: mul24_test:
; CHECK:       mul
; CHECK:       halt
  %x = load i32, ptr addrspace(1) %a, align 4
  %y = load i32, ptr addrspace(1) %b, align 4
  %p = call spir_func i32 @_Z5mul24ii(i32 %x, i32 %y)
  store i32 %p, ptr addrspace(1) %out, align 4
  ret void
}

define dso_local spir_kernel void @mad24_test(ptr addrspace(1) %a,
                                              ptr addrspace(1) %b,
                                              ptr addrspace(1) %c,
                                              ptr addrspace(1) %out) {
; CHECK-LABEL: mad24_test:
; CHECK:       mul
; CHECK:       add
; CHECK:       halt
  %x = load i32, ptr addrspace(1) %a, align 4
  %y = load i32, ptr addrspace(1) %b, align 4
  %z = load i32, ptr addrspace(1) %c, align 4
  %p = call spir_func i32 @_Z5mad24iii(i32 %x, i32 %y, i32 %z)
  store i32 %p, ptr addrspace(1) %out, align 4
  ret void
}
