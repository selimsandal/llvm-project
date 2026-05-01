; RUN: not llc -march=gpu -mtriple=gpu-none-none %s -o /dev/null 2>&1 | FileCheck %s

target triple = "dxilv1.0-pc-shadermodel6.0-compute"

declare hidden void @_Z12GPUEmitPixelfff(float, float, float) #1
declare hidden void @_Z12GPUEmitPixeliii(i32, i32, i32) #1
declare hidden void @_Z12GPUEmitPixeljj(i32, i32) #1
declare hidden i32 @_Z12GPUEmitPixeljjj(i32, i32, i32) #1

define void @GPUEmitPixel(i32 %offset, i32 %depth, i32 %color) #2 {
entry:
  ret void
}

define void @bad_emit_pixel(float %offset, float %depth, float %color) #0 {
entry:
  call void @_Z12GPUEmitPixelfff(float %offset, float %depth, float %color)
  ret void
}

define void @bad_emit_pixel_arg_count(i32 %offset, i32 %depth) #0 {
entry:
  call void @_Z12GPUEmitPixeljj(i32 %offset, i32 %depth)
  ret void
}

define void @bad_emit_pixel_signed_overload(i32 %offset, i32 %depth,
                                            i32 %color) #0 {
entry:
  call void @_Z12GPUEmitPixeliii(i32 %offset, i32 %depth, i32 %color)
  ret void
}

define i32 @bad_emit_pixel_return_type(i32 %offset, i32 %depth, i32 %color) #0 {
entry:
  %ret = call i32 @_Z12GPUEmitPixeljjj(i32 %offset, i32 %depth, i32 %color)
  ret i32 %ret
}

define void @bad_emit_pixel_definition(i32 %offset, i32 %depth,
                                       i32 %color) #0 {
entry:
  call void @GPUEmitPixel(i32 %offset, i32 %depth, i32 %color)
  ret void
}

attributes #0 = { convergent noinline norecurse "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }
attributes #1 = { convergent }
attributes #2 = { convergent noinline optnone }

; CHECK: error: {{.*}}GPUEmitPixel must be declared as void GPUEmitPixel(uint offset, uint depth, uint color)
; CHECK: error: {{.*}}GPUEmitPixel must be declared as void GPUEmitPixel(uint offset, uint depth, uint color)
; CHECK: error: {{.*}}GPUEmitPixel must be declared as void GPUEmitPixel(uint offset, uint depth, uint color)
; CHECK: error: {{.*}}GPUEmitPixel must be declared as void GPUEmitPixel(uint offset, uint depth, uint color)
; CHECK: error: {{.*}}GPUEmitPixel must be declared as void GPUEmitPixel(uint offset, uint depth, uint color)
