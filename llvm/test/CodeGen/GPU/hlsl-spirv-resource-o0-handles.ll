; RUN: llc -march=gpu -mtriple=gpu-none-none -filetype=obj %s -o %t.o
; RUN: llc -march=gpu -mtriple=gpu-none-none %s -o - | FileCheck %s
; RUN: llc -march=gpu -mtriple=gpu-none-none -print-after=gpu-spirv-lowering %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=IR

target triple = "spirv-unknown-vulkan-compute"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-p11:32:32-i32:32-f32:32-n32"

%"class.hlsl::RWStructuredBuffer" = type { target("spirv.VulkanBuffer", [0 x i32], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }

@Out = internal global %"class.hlsl::RWStructuredBuffer" poison, align 8
@out.str = private unnamed_addr constant [4 x i8] c"Out\00"

declare target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32, i32, i32, i32, ptr)
declare target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefromimplicitbinding.tspirv.VulkanBuffer_i32_12_1t.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1), i32, i32)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1), i32)
declare i32 @llvm.spv.thread.id.in.group.i32(i32)

define void @CSMain() #0 {
; CHECK-LABEL: CSMain:
; CHECK: getsr {{r[0-9]+}}, 59
; CHECK: add {{r[0-9]+}}, r1, {{r[0-9]+}}
; CHECK: st_scatter
; CHECK: halt
;
; IR-LABEL: IR Dump After GPU SPIR-V Lowering
; IR-NOT: llvm.spv.resource
; IR-NOT: load target("spirv.VulkanBuffer"
; IR-NOT: store target("spirv.VulkanBuffer"
entry:
  %bind.addr = alloca i32, align 4
  %tmp = alloca %"class.hlsl::RWStructuredBuffer", align 8
  store i32 0, ptr %bind.addr, align 4
  %bind = load i32, ptr %bind.addr, align 4
  %handle = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 %bind, i32 1, i32 0, ptr @out.str)
  store target("spirv.VulkanBuffer", [0 x i32], 12, 1) %handle, ptr %tmp, align 8
  %handle.for.counter = load target("spirv.VulkanBuffer", [0 x i32], 12, 1), ptr %tmp, align 8
  %counter = call target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefromimplicitbinding.tspirv.VulkanBuffer_i32_12_1t.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %handle.for.counter, i32 0, i32 0)
  %counter.ptr = getelementptr inbounds %"class.hlsl::RWStructuredBuffer", ptr %tmp, i32 0, i32 1
  store target("spirv.VulkanBuffer", i32, 12, 1) %counter, ptr %counter.ptr, align 8
  %handle.copy = load target("spirv.VulkanBuffer", [0 x i32], 12, 1), ptr %tmp, align 8
  store target("spirv.VulkanBuffer", [0 x i32], 12, 1) %handle.copy, ptr @Out, align 8
  %id = call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %loaded = load target("spirv.VulkanBuffer", [0 x i32], 12, 1), ptr @Out, align 8
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %loaded, i32 %id)
  store i32 %id, ptr addrspace(11) %ptr, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }
