; RUN: llc -march=gpu %s -o - | FileCheck %s

; Test the target intrinsic that exposes the RTL ROP path to LLVM IR.
; Operand order matches the GPU ISA:
;   pixel_out offset, depth, color

target triple = "gpu-none-none"
target datalayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32"

declare void @llvm.gpu.pixel.out(i32, i32, i32)

define void @emit_pixel(i32 %offset, i32 %depth, i32 %color) {
entry:
  call void @llvm.gpu.pixel.out(i32 %offset, i32 %depth, i32 %color)
  ret void
}

; CHECK-LABEL: emit_pixel:
; CHECK: pixel_out{{[[:space:]]+}}r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
; CHECK: halt

define void @emit_pixel_constants() {
entry:
  call void @llvm.gpu.pixel.out(i32 32, i32 1065353216, i32 4278255360)
  ret void
}

; CHECK-LABEL: emit_pixel_constants:
; CHECK-DAG: movi{{[[:space:]]+}}r{{[0-9]+}}, 0x20
; CHECK-DAG: movi{{[[:space:]]+}}r{{[0-9]+}}, 0x3f800000
; CHECK-DAG: movi{{[[:space:]]+}}r{{[0-9]+}}, 0x{{f+}}00ff00
; CHECK: pixel_out{{[[:space:]]+}}r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
; CHECK: halt

define void @emit_pixel_computed(i32 %base, i32 %depth_bits, i32 %color_bits) {
entry:
  %offset = shl i32 %base, 2
  %depth = xor i32 %depth_bits, 255
  %color = or i32 %color_bits, 4278190080
  call void @llvm.gpu.pixel.out(i32 %offset, i32 %depth, i32 %color)
  ret void
}

; CHECK-LABEL: emit_pixel_computed:
; CHECK-DAG: shl
; CHECK-DAG: xor
; CHECK-DAG: or
; CHECK: pixel_out{{[[:space:]]+}}r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
; CHECK: halt

define void @emit_two_pixels(i32 %offset, i32 %depth, i32 %color) {
entry:
  call void @llvm.gpu.pixel.out(i32 %offset, i32 %depth, i32 %color)
  %next_offset = add i32 %offset, 4
  call void @llvm.gpu.pixel.out(i32 %next_offset, i32 %depth, i32 %color)
  ret void
}

; CHECK-LABEL: emit_two_pixels:
; CHECK: pixel_out{{[[:space:]]+}}r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
; CHECK: add
; CHECK: pixel_out{{[[:space:]]+}}r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
; CHECK: halt

define void @emit_pixel_conditional(i32 %offset, i32 %depth, i32 %color,
                                    i32 %enable) {
entry:
  %do_emit = icmp ne i32 %enable, 0
  br i1 %do_emit, label %emit, label %exit

emit:
  call void @llvm.gpu.pixel.out(i32 %offset, i32 %depth, i32 %color)
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: emit_pixel_conditional:
; CHECK: cmp
; CHECK: goto
; CHECK: pixel_out{{[[:space:]]+}}r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
; CHECK: halt
