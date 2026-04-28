; RUN: llc -march=gpu -mtriple=gpu-none-none -filetype=obj %s -o %t.o
; RUN: llc -march=gpu -mtriple=gpu-none-none %s -o - | FileCheck %s

target triple = "spirv-unknown-vulkan-compute"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-p11:32:32-i32:32-f32:32-n32"

@buf4.str = private unnamed_addr constant [5 x i8] c"Buf4\00"

declare target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4i32_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 1), i32)
declare i32 @llvm.spv.thread.id.in.group.i32(i32)

define void @store_bind4_v4i32_indirect() #0 {
; CHECK-LABEL: store_bind4_v4i32_indirect:
; CHECK: ld_scatter {{r[0-9]+}}, [r1 + 0x10]
; CHECK: getsr {{r[0-9]+}}, 59
; CHECK: shl {{r[0-9]+}}, {{r[0-9]+}}, 0x4
; CHECK: st_scatter
; CHECK: halt
entry:
  %handle = call target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4i32_12_1t(i32 0, i32 4, i32 1, i32 0, ptr @buf4.str)
  %id = call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 1) %handle, i32 %id)
  store i32 %id, ptr addrspace(11) %ptr, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }
