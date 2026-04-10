; RUN: llc -march=gpu -O2 -filetype=asm %s -o - | FileCheck %s

; This kernel has a real if/else near the end, but one arm originally ended in
; a direct HALT while the other fell through a shared pure-exit block. The
; control-flow lowering must still preserve a split shape, with a second GOTO
; that skips the else body after the then body completes.

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir"

; CHECK-LABEL: gpu_cf_repro:
; CHECK:       add r11, r11, r7
; CHECK-NEXT:  ld_scatter [[CUR:r[0-9]+]], [r11 + 0x8]
; CHECK-NEXT:  cmp [[XCOST:r[0-9]+]], [[CUR]]
; CHECK-NEXT:  goto f0, {{[0-9]+}}
; CHECK:       st_scatter [{{r[0-9]+}} + 0x0], {{r[0-9]+}}
; CHECK:       join
; CHECK-NEXT:  goto f0, {{[0-9]+}}
; CHECK:       ld_scatter {{r[0-9]+}}, [{{r[0-9]+}} + 0x4]
; CHECK:       st_scatter [{{r[0-9]+}} + 0x0], {{r[0-9]+}}
; CHECK:       join
; CHECK:       join
; CHECK:       halt

define dso_local spir_kernel void @gpu_cf_repro(ptr addrspace(1) noundef readonly align 4 captures(none) %0,
                                                ptr addrspace(1) noundef readonly align 4 captures(none) %1,
                                                ptr addrspace(1) noundef align 4 captures(none) %2,
                                                ptr addrspace(1) noundef readonly align 4 captures(none) %3,
                                                ptr addrspace(1) noundef writeonly align 4 captures(none) %4,
                                                ptr addrspace(3) noundef align 4 captures(none) %5,
                                                i32 noundef %6, i32 noundef %7,
                                                i32 noundef %8, i32 noundef %9)
    local_unnamed_addr !kernel_arg_addr_space !0
    !kernel_arg_access_qual !1
    !kernel_arg_type !2
    !kernel_arg_base_type !2
    !kernel_arg_type_qual !3 {
  %11 = call spir_func i32 @_Z13get_global_idj(i32 0)
  %12 = call spir_func i32 @_Z12get_local_idj(i32 0)
  %13 = icmp slt i32 %11, %6
  br i1 %13, label %14, label %66

14:
  %15 = icmp eq i32 %12, 0
  %16 = icmp sgt i32 %7, 0
  %17 = and i1 %16, %15
  br i1 %17, label %18, label %28

18:
  %19 = getelementptr [4 x i8], ptr addrspace(1) %1, i32 %8
  br label %20

20:
  %21 = phi i32 [ 0, %18 ], [ %26, %20 ]
  %22 = mul nsw i32 %21, %6
  %23 = getelementptr [4 x i8], ptr addrspace(1) %19, i32 %22
  %24 = load float, ptr addrspace(1) %23, align 4
  %25 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %5, i32 %21
  store float %24, ptr addrspace(3) %25, align 4
  %26 = add nuw nsw i32 %21, 1
  %27 = icmp slt i32 %26, %7
  br i1 %27, label %20, label %28

28:
  call spir_func void @_Z7barrierj(i32 1)
  br i1 %16, label %29, label %43

29:
  %30 = getelementptr [4 x i8], ptr addrspace(1) %1, i32 %11
  br label %31

31:
  %32 = phi float [ 0.0, %29 ], [ %40, %31 ]
  %33 = phi i32 [ 0, %29 ], [ %41, %31 ]
  %34 = mul nsw i32 %33, %6
  %35 = getelementptr [4 x i8], ptr addrspace(1) %30, i32 %34
  %36 = load float, ptr addrspace(1) %35, align 4
  %37 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %5, i32 %33
  %38 = load float, ptr addrspace(3) %37, align 4
  %39 = fsub float %36, %38
  %40 = call float @llvm.fmuladd.f32(float %39, float %39, float %32)
  %41 = add nuw nsw i32 %33, 1
  %42 = icmp slt i32 %41, %7
  br i1 %42, label %31, label %43

43:
  %44 = phi float [ 0.0, %28 ], [ %40, %31 ]
  %45 = getelementptr inbounds [12 x i8], ptr addrspace(1) %0, i32 %11
  %46 = getelementptr inbounds nuw i8, ptr addrspace(1) %45, i32 8
  %47 = load float, ptr addrspace(1) %46, align 4
  %48 = add nsw i32 %9, 1
  %49 = mul nsw i32 %11, %48
  %50 = fcmp olt float %44, %47
  br i1 %50, label %51, label %56

51:
  %52 = getelementptr inbounds [4 x i8], ptr addrspace(1) %4, i32 %11
  store i32 1, ptr addrspace(1) %52, align 4
  %53 = fsub float %44, %47
  %54 = getelementptr [4 x i8], ptr addrspace(1) %2, i32 %49
  %55 = getelementptr [4 x i8], ptr addrspace(1) %54, i32 %9
  store float %53, ptr addrspace(1) %55, align 4
  br label %66

56:
  %57 = getelementptr inbounds nuw i8, ptr addrspace(1) %45, i32 4
  %58 = load i32, ptr addrspace(1) %57, align 4
  %59 = getelementptr inbounds [4 x i8], ptr addrspace(1) %3, i32 %58
  %60 = load i32, ptr addrspace(1) %59, align 4
  %61 = fsub float %47, %44
  %62 = getelementptr [4 x i8], ptr addrspace(1) %2, i32 %49
  %63 = getelementptr [4 x i8], ptr addrspace(1) %62, i32 %60
  %64 = load float, ptr addrspace(1) %63, align 4
  %65 = fadd float %61, %64
  store float %65, ptr addrspace(1) %63, align 4
  br label %66

66:
  ret void
}

declare spir_func i32 @_Z13get_global_idj(i32)
declare spir_func i32 @_Z12get_local_idj(i32)
declare spir_func void @_Z7barrierj(i32)
declare float @llvm.fmuladd.f32(float, float, float)

!0 = !{i32 1, i32 1, i32 1, i32 1, i32 1, i32 3, i32 0, i32 0, i32 0, i32 0}
!1 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!2 = !{!"Point32*", !"float*", !"float*", !"int*", !"int*", !"float*", !"int", !"int", !"int", !"int"}
!3 = !{!"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
