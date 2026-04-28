; RUN: llc -march=gpu -mtriple=gpu-none-none -filetype=obj %s -o %t.o
; RUN: llc -march=gpu -mtriple=gpu-none-none %s -o - | FileCheck %s

target triple = "spirv-unknown-vulkan-compute"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-p11:32:32-i32:32-f32:32-n32"

@out.str = private unnamed_addr constant [4 x i8] c"Out\00"
@in.str = private unnamed_addr constant [3 x i8] c"In\00"
@buf2.str = private unnamed_addr constant [5 x i8] c"Buf2\00"

declare target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1), i32)
declare target("spirv.VulkanBuffer", [0 x i32], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_0t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_0t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 0), i32)
declare target("spirv.VulkanBuffer", [0 x i16], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i16_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i16_12_1t.i32(target("spirv.VulkanBuffer", [0 x i16], 12, 1), i32)
declare i32 @llvm.spv.thread.id.in.group.i32(i32)

define void @CSMain() #0 {
; CHECK-LABEL: CSMain:
; CHECK: getsr {{r[0-9]+}}, 59
; CHECK: st_scatter
; CHECK: halt
entry:
  %handle = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr @out.str)
  %id = call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %handle, i32 %id)
  store i32 %id, ptr addrspace(11) %ptr, align 4
  ret void
}

define void @load_readonly_bind1() #0 {
; CHECK-LABEL: load_readonly_bind1:
; CHECK: getsr {{r[0-9]+}}, 59
; CHECK: add {{r[0-9]+}}, r2, {{r[0-9]+}}
; CHECK: ld_scatter
; CHECK: st_scatter
; CHECK: halt
entry:
  %in = call target("spirv.VulkanBuffer", [0 x i32], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_0t(i32 0, i32 1, i32 1, i32 0, ptr @in.str)
  %out = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr @out.str)
  %id = call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %in.ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_0t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 0) %in, i32 %id)
  %val = load i32, ptr addrspace(11) %in.ptr, align 4
  %out.ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %out, i32 %id)
  store i32 %val, ptr addrspace(11) %out.ptr, align 4
  ret void
}

define void @store_bind2_i16_scaled() #0 {
; CHECK-LABEL: store_bind2_i16_scaled:
; CHECK: getsr {{r[0-9]+}}, 59
; CHECK: shl {{r[0-9]+}}, {{r[0-9]+}}, 0x1
; CHECK: add {{r[0-9]+}}, r3, {{r[0-9]+}}
; CHECK: st_scatter
; CHECK: halt
entry:
  %handle = call target("spirv.VulkanBuffer", [0 x i16], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i16_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @buf2.str)
  %id = call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i16_12_1t.i32(target("spirv.VulkanBuffer", [0 x i16], 12, 1) %handle, i32 %id)
  store i32 %id, ptr addrspace(11) %ptr, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }
