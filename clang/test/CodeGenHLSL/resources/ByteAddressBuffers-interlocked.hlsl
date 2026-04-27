// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

RWByteAddressBuffer Counter : register(u0);

export void Test(uint idx) {
  uint old;
  Counter.InterlockedAdd(idx, 1, old);
  Counter.InterlockedAnd(idx, 255, old);
  Counter.InterlockedOr(idx, 1, old);
  Counter.InterlockedXor(idx, 1, old);
  Counter.InterlockedMin(idx, 7, old);
  Counter.InterlockedMax(idx, 9, old);
  Counter.InterlockedExchange(idx, 11, old);
  Counter.InterlockedCompareExchange(idx, 0, 42, old);
}

export void TestNoOriginal(uint idx) {
  Counter.InterlockedAdd(idx, 1);
  Counter.InterlockedExchange(idx, 2);
}

// CHECK-LABEL: define linkonce_odr hidden void @_ZN4hlsl19RWByteAddressBuffer14InterlockedAddEjjRj
// CHECK: call ptr @llvm.dx.resource.getpointer
// CHECK: atomicrmw add ptr

// CHECK-LABEL: define linkonce_odr hidden void @_ZN4hlsl19RWByteAddressBuffer14InterlockedAndEjjRj
// CHECK: atomicrmw and ptr

// CHECK-LABEL: define linkonce_odr hidden void @_ZN4hlsl19RWByteAddressBuffer13InterlockedOrEjjRj
// CHECK: atomicrmw or ptr

// CHECK-LABEL: define linkonce_odr hidden void @_ZN4hlsl19RWByteAddressBuffer14InterlockedXorEjjRj
// CHECK: atomicrmw xor ptr

// CHECK-LABEL: define linkonce_odr hidden void @_ZN4hlsl19RWByteAddressBuffer14InterlockedMinEjjRj
// CHECK: atomicrmw umin ptr

// CHECK-LABEL: define linkonce_odr hidden void @_ZN4hlsl19RWByteAddressBuffer14InterlockedMaxEjjRj
// CHECK: atomicrmw umax ptr

// CHECK-LABEL: define linkonce_odr hidden void @_ZN4hlsl19RWByteAddressBuffer19InterlockedExchangeEjjRj
// CHECK: atomicrmw xchg ptr

// CHECK-LABEL: define linkonce_odr hidden void @_ZN4hlsl19RWByteAddressBuffer26InterlockedCompareExchangeEjjjRj
// CHECK: cmpxchg ptr

// CHECK-LABEL: define linkonce_odr hidden void @_ZN4hlsl19RWByteAddressBuffer14InterlockedAddEjj
// CHECK: atomicrmw add ptr

// CHECK-LABEL: define linkonce_odr hidden void @_ZN4hlsl19RWByteAddressBuffer19InterlockedExchangeEjj
// CHECK: atomicrmw xchg ptr
