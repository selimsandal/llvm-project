// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -check-prefixes=CHECK,CHECK-SPIRV

void test_BarrierVariants() {
// CHECK-DXIL: call void @llvm.dx.group.memory.barrier()
// CHECK-SPIRV: call spir_func void @llvm.spv.group.memory.barrier()
  GroupMemoryBarrier();

// CHECK-DXIL: call void @llvm.dx.device.memory.barrier()
// CHECK-SPIRV: call spir_func void @llvm.spv.device.memory.barrier()
  DeviceMemoryBarrier();

// CHECK-DXIL: call void @llvm.dx.all.memory.barrier()
// CHECK-SPIRV: call spir_func void @llvm.spv.all.memory.barrier()
  AllMemoryBarrier();

// CHECK-DXIL: call void @llvm.dx.group.memory.barrier.with.group.sync()
// CHECK-SPIRV: call spir_func void @llvm.spv.group.memory.barrier.with.group.sync()
  GroupMemoryBarrierWithGroupSync();

// CHECK-DXIL: call void @llvm.dx.device.memory.barrier.with.group.sync()
// CHECK-SPIRV: call spir_func void @llvm.spv.device.memory.barrier.with.group.sync()
  DeviceMemoryBarrierWithGroupSync();

// CHECK-DXIL: call void @llvm.dx.all.memory.barrier.with.group.sync()
// CHECK-SPIRV: call spir_func void @llvm.spv.all.memory.barrier.with.group.sync()
  AllMemoryBarrierWithGroupSync();
}

// CHECK: declare {{.*}}void @llvm.{{(dx|spv)}}.group.memory.barrier()
// CHECK: declare {{.*}}void @llvm.{{(dx|spv)}}.device.memory.barrier()
// CHECK: declare {{.*}}void @llvm.{{(dx|spv)}}.all.memory.barrier()
// CHECK: declare {{.*}}void @llvm.{{(dx|spv)}}.group.memory.barrier.with.group.sync()
// CHECK: declare {{.*}}void @llvm.{{(dx|spv)}}.device.memory.barrier.with.group.sync()
// CHECK: declare {{.*}}void @llvm.{{(dx|spv)}}.all.memory.barrier.with.group.sync()
