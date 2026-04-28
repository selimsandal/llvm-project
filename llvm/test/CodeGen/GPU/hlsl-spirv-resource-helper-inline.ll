; RUN: llc -march=gpu -mtriple=gpu-none-none -filetype=obj %s -o %t.o
; RUN: llc -march=gpu -mtriple=gpu-none-none %s -o - | FileCheck %s
; RUN: llc -march=gpu -mtriple=gpu-none-none -print-after=gpu-spirv-lowering %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=IR

target triple = "spirv-unknown-vulkan-compute"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-p11:32:32-i32:32-f32:32-n32"

%"class.hlsl::RWStructuredBuffer" = type { target("spirv.VulkanBuffer", [0 x i32], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }

@Out = internal global %"class.hlsl::RWStructuredBuffer" poison, align 8
@out.str = private unnamed_addr constant [4 x i8] c"Out\00"

declare target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1), i32)
declare i32 @llvm.spv.thread.id.in.group.i32(i32)

define internal void @init_out() #0 {
entry:
  %tmp = alloca %"class.hlsl::RWStructuredBuffer", align 8
  %handle = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr @out.str)
  store target("spirv.VulkanBuffer", [0 x i32], 12, 1) %handle, ptr %tmp, align 8
  %loaded = load target("spirv.VulkanBuffer", [0 x i32], 12, 1), ptr %tmp, align 8
  store target("spirv.VulkanBuffer", [0 x i32], 12, 1) %loaded, ptr @Out, align 8
  ret void
}

define internal ptr addrspace(11) @index_out(ptr %this, i32 %idx) #0 {
entry:
  %handle.ptr = getelementptr inbounds %"class.hlsl::RWStructuredBuffer", ptr %this, i32 0, i32 0
  %handle = load target("spirv.VulkanBuffer", [0 x i32], 12, 1), ptr %handle.ptr, align 8
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %handle, i32 %idx)
  ret ptr addrspace(11) %ptr
}

define void @CSMain() #1 {
; CHECK-LABEL: CSMain:
; CHECK: getsr {{r[0-9]+}}, 59
; CHECK: add {{r[0-9]+}}, r1, {{r[0-9]+}}
; CHECK: st_scatter
; CHECK: halt
;
; IR-LABEL: IR Dump After GPU SPIR-V Lowering
; IR-NOT: @init_out
; IR-NOT: @index_out
; IR-NOT: llvm.spv.resource
; IR-NOT: spirv.VulkanBuffer
entry:
  call void @init_out()
  %id = call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %ptr = call ptr addrspace(11) @index_out(ptr @Out, i32 %id)
  store i32 %id, ptr addrspace(11) %ptr, align 4
  ret void
}

attributes #0 = { alwaysinline nounwind }
attributes #1 = { "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }
