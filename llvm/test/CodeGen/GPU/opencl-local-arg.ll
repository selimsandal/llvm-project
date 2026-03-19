; RUN: llc -march=gpu < %s | FileCheck %s

target triple = "spir"

declare dso_local spir_func i32 @_Z12get_local_idj(i32 noundef)
declare dso_local spir_func i32 @_Z13get_global_idj(i32 noundef)
declare dso_local spir_func void @_Z7barrierj(i32 noundef)

define dso_local spir_kernel void @k(ptr addrspace(3) %scratch,
                                     ptr addrspace(1) %out) {
entry:
  %lid = call spir_func i32 @_Z12get_local_idj(i32 0)
  %sptr = getelementptr inbounds i32, ptr addrspace(3) %scratch, i32 %lid
  store i32 %lid, ptr addrspace(3) %sptr, align 4
  call spir_func void @_Z7barrierj(i32 1)
  %next = add i32 %lid, 1
  %next.masked = and i32 %next, 7
  %lptr = getelementptr inbounds i32, ptr addrspace(3) %scratch, i32 %next.masked
  %val = load i32, ptr addrspace(3) %lptr, align 4
  %gid = call spir_func i32 @_Z13get_global_idj(i32 0)
  %out.ptr = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %gid
  store i32 %val, ptr addrspace(1) %out.ptr, align 4
  ret void
}

; CHECK-LABEL: k:
; CHECK: getsr
; CHECK: add
; CHECK: st_local
; CHECK: barrier	1
; CHECK: ld_local
; CHECK: st_scatter
; CHECK: halt
