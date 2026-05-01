; RUN: llc -march=gpu -mtriple=gpu-none-none %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -march=gpu -mtriple=gpu-none-none -filetype=obj %s -o /dev/null
; RUN: llc -march=gpu -mtriple=gpu-none-none -print-after=gpu-hlsl-lowering %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=IR

; Test: HLSL source-level GPUEmitPixel(offset, depth, color) lowering.
;
; Equivalent source shape:
;   void GPUEmitPixel(uint offset, uint depth, uint color);
;   [numthreads(8, 1, 1)]
;   void main(uint3 tid : SV_DispatchThreadID) {
;     GPUEmitPixel(tid.x * 4, 16 + tid.x, 0xAA000000u | tid.x);
;   }

target triple = "dxilv1.0-pc-shadermodel6.0-compute"

declare i32 @llvm.dx.thread.id(i32)
declare hidden void @_Z12GPUEmitPixeljjj(i32, i32, i32) #1
declare void @GPUEmitPixel(i32, i32, i32) #1

define void @hlsl_emit_pixel() #0 {
; ASM-LABEL: hlsl_emit_pixel:
; ASM: getsr
; ASM: pixel_out
; ASM: cmp
; ASM: goto
; ASM: pixel_out
; ASM: halt
;
; IR-LABEL: define void @hlsl_emit_pixel
; IR-NOT: GPUEmitPixel
; IR: call void @llvm.gpu.pixel.out(i32 %offset, i32 %depth, i32 %color)
; IR: call void @llvm.gpu.pixel.out(i32 %off2, i32 1, i32 -16711936)
entry:
  %tid = call i32 @llvm.dx.thread.id(i32 0)
  %offset = shl i32 %tid, 2
  %depth = add i32 %tid, 16
  %color = or i32 %tid, -1442840576
  call void @_Z12GPUEmitPixeljjj(i32 %offset, i32 %depth, i32 %color)

  %lane.bit = and i32 %tid, 1
  %do.emit = icmp eq i32 %lane.bit, 0
  br i1 %do.emit, label %emit2, label %exit

emit2:
  %off2 = add i32 %offset, 32
  call void @_Z12GPUEmitPixeljjj(i32 %off2, i32 1, i32 -16711936)
  br label %exit

exit:
  ret void
}

attributes #0 = { convergent noinline norecurse "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }
attributes #1 = { convergent }

define void @hlsl_emit_pixel_exact_name(i32 %offset, i32 %depth,
                                        i32 %color) #0 {
; ASM-LABEL: hlsl_emit_pixel_exact_name:
; ASM: pixel_out
; ASM: halt
;
; IR-LABEL: define void @hlsl_emit_pixel_exact_name
; IR-NOT: GPUEmitPixel
; IR: call void @llvm.gpu.pixel.out(i32 %offset, i32 %depth, i32 %color)
entry:
  call void @GPUEmitPixel(i32 %offset, i32 %depth, i32 %color)
  ret void
}
