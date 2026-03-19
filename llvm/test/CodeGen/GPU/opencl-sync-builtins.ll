; RUN: llc -march=gpu < %s | FileCheck %s

target triple = "spir"

declare dso_local spir_func i32 @_Z13get_global_idj(i32 noundef)
declare dso_local spir_func i32 @_Z12get_local_idj(i32 noundef)
declare dso_local spir_func void @_Z7barrierj(i32 noundef)
declare dso_local spir_func void @_Z9mem_fencej(i32 noundef)

define dso_local spir_kernel void @kernel(ptr addrspace(1) %out) {
entry:
  call spir_func void @_Z7barrierj(i32 1)
  call spir_func void @_Z9mem_fencej(i32 2)
  %gid = call spir_func i32 @_Z13get_global_idj(i32 0)
  %lid = call spir_func i32 @_Z12get_local_idj(i32 0)
  %out.ptr = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %gid
  store i32 %lid, ptr addrspace(1) %out.ptr, align 4
  ret void
}

; CHECK-LABEL: kernel:
; CHECK: getsr
; CHECK: barrier	1
; CHECK: mem_fence	2
; CHECK: st_scatter
; CHECK: halt
