; RUN: llc -march=gpu < %s | FileCheck %s

target triple = "spir"

declare dso_local spir_func i32 @_Z10atomic_addPU3AS3Vii(ptr addrspace(3), i32)
declare dso_local spir_func i32 @_Z11atomic_xchgPU3AS3Vii(ptr addrspace(3), i32)
declare dso_local spir_func i32 @_Z9atomic_orPU3AS3Vii(ptr addrspace(3), i32)
declare dso_local spir_func i32 @_Z10atomic_andPU3AS3Vii(ptr addrspace(3), i32)
declare dso_local spir_func i32 @_Z10atomic_xorPU3AS3Vii(ptr addrspace(3), i32)
declare dso_local spir_func i32 @_Z10atomic_minPU3AS3Vii(ptr addrspace(3), i32)
declare dso_local spir_func i32 @_Z10atomic_maxPU3AS3Vii(ptr addrspace(3), i32)
declare dso_local spir_func i32 @_Z10atomic_minPU3AS3Vjj(ptr addrspace(3), i32)
declare dso_local spir_func i32 @_Z10atomic_maxPU3AS3Vjj(ptr addrspace(3), i32)
declare dso_local spir_func i32 @_Z14atomic_cmpxchgPU3AS3Viii(ptr addrspace(3), i32, i32)

define dso_local spir_kernel void @k(ptr addrspace(3) %scratch,
                                     ptr addrspace(1) %out,
                                     i32 %v,
                                     i32 %cmp) {
entry:
  %old0 = call spir_func i32 @_Z10atomic_addPU3AS3Vii(ptr addrspace(3) %scratch,
                                                      i32 %v)
  %p1 = getelementptr inbounds i32, ptr addrspace(3) %scratch, i32 1
  %old1 = call spir_func i32 @_Z11atomic_xchgPU3AS3Vii(ptr addrspace(3) %p1,
                                                       i32 %v)
  %p2 = getelementptr inbounds i32, ptr addrspace(3) %scratch, i32 2
  %old2 = call spir_func i32 @_Z9atomic_orPU3AS3Vii(ptr addrspace(3) %p2,
                                                    i32 %v)
  %p3 = getelementptr inbounds i32, ptr addrspace(3) %scratch, i32 3
  %old3 = call spir_func i32 @_Z10atomic_andPU3AS3Vii(ptr addrspace(3) %p3,
                                                      i32 %v)
  %p4 = getelementptr inbounds i32, ptr addrspace(3) %scratch, i32 4
  %old4 = call spir_func i32 @_Z10atomic_xorPU3AS3Vii(ptr addrspace(3) %p4,
                                                      i32 %v)
  %p5 = getelementptr inbounds i32, ptr addrspace(3) %scratch, i32 5
  %old5 = call spir_func i32 @_Z10atomic_minPU3AS3Vii(ptr addrspace(3) %p5,
                                                      i32 %v)
  %p6 = getelementptr inbounds i32, ptr addrspace(3) %scratch, i32 6
  %old6 = call spir_func i32 @_Z10atomic_maxPU3AS3Vii(ptr addrspace(3) %p6,
                                                      i32 %v)
  %p7 = getelementptr inbounds i32, ptr addrspace(3) %scratch, i32 7
  %old7 = call spir_func i32 @_Z14atomic_cmpxchgPU3AS3Viii(ptr addrspace(3) %p7,
                                                           i32 %cmp, i32 %v)
  %p8 = getelementptr inbounds i32, ptr addrspace(3) %scratch, i32 8
  %old8 = call spir_func i32 @_Z10atomic_minPU3AS3Vjj(ptr addrspace(3) %p8,
                                                      i32 %v)
  %p9 = getelementptr inbounds i32, ptr addrspace(3) %scratch, i32 9
  %old9 = call spir_func i32 @_Z10atomic_maxPU3AS3Vjj(ptr addrspace(3) %p9,
                                                      i32 %v)
  %s0 = add i32 %old0, %old1
  %s1 = add i32 %s0, %old2
  %s2 = add i32 %s1, %old3
  %s3 = add i32 %s2, %old4
  %s4 = add i32 %s3, %old5
  %s5 = add i32 %s4, %old6
  %s6 = add i32 %s5, %old7
  %s7 = add i32 %s6, %old8
  %sum = add i32 %s7, %old9
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: k:
; CHECK-COUNT-7: atomic_local{{\t| }}
; CHECK: atomic_local_cas
; CHECK-COUNT-2: atomic_local{{\t| }}
; CHECK: st_scatter
; CHECK: halt
