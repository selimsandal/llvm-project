; RUN: llc -march=gpu < %s | FileCheck %s

target triple = "spir"

define dso_local spir_kernel void @k(ptr addrspace(3) noundef align 4 %0,
                                     ptr addrspace(1) noundef align 4 %1) #0 {
entry:
  %3 = alloca ptr addrspace(3), align 4
  %4 = alloca ptr addrspace(1), align 4
  store ptr addrspace(3) %0, ptr %3, align 4
  store ptr addrspace(1) %1, ptr %4, align 4
  %5 = load ptr addrspace(3), ptr %3, align 4
  %6 = load ptr addrspace(1), ptr %4, align 4
  call spir_func void @__clang_ocl_kern_imp_k(ptr addrspace(3) noundef align 4 %5,
                                              ptr addrspace(1) noundef align 4 %6) #3
  ret void
}

define dso_local spir_func void @__clang_ocl_kern_imp_k(ptr addrspace(3) noundef align 4 %0,
                                                        ptr addrspace(1) noundef align 4 %1) #0 {
entry:
  %3 = alloca ptr addrspace(3), align 4
  %4 = alloca ptr addrspace(1), align 4
  %5 = alloca i32, align 4
  store ptr addrspace(3) %0, ptr %3, align 4
  store ptr addrspace(1) %1, ptr %4, align 4
  %6 = call spir_func i32 @_Z12get_local_idj(i32 noundef 0) #4
  store i32 %6, ptr %5, align 4
  %7 = load i32, ptr %5, align 4
  %8 = load ptr addrspace(3), ptr %3, align 4
  %9 = load i32, ptr %5, align 4
  %10 = getelementptr inbounds i32, ptr addrspace(3) %8, i32 %9
  store i32 %7, ptr addrspace(3) %10, align 4
  call spir_func void @_Z7barrierj(i32 noundef 1) #3
  %11 = load ptr addrspace(3), ptr %3, align 4
  %12 = load i32, ptr %5, align 4
  %13 = add nsw i32 %12, 1
  %14 = and i32 %13, 7
  %15 = getelementptr inbounds i32, ptr addrspace(3) %11, i32 %14
  %16 = load i32, ptr addrspace(3) %15, align 4
  %17 = load ptr addrspace(1), ptr %4, align 4
  %18 = load i32, ptr %5, align 4
  %19 = getelementptr inbounds i32, ptr addrspace(1) %17, i32 %18
  store i32 %16, ptr addrspace(1) %19, align 4
  ret void
}

declare dso_local spir_func i32 @_Z12get_local_idj(i32 noundef) #1
declare dso_local spir_func void @_Z7barrierj(i32 noundef) #2

attributes #0 = { convergent noinline norecurse nounwind optnone }
attributes #1 = { convergent nounwind willreturn memory(none) }
attributes #2 = { convergent nounwind }
attributes #3 = { convergent nounwind }
attributes #4 = { convergent nounwind willreturn memory(none) }

; CHECK-LABEL: k:
; CHECK: getsr
; CHECK: st_local
; CHECK: barrier
; CHECK: ld_local
; CHECK: st_scatter
; CHECK: halt
