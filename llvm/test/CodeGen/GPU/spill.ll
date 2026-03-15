; RUN: llc -march=gpu -mtriple=gpu-none-none < %s | FileCheck %s
; Test that the pathtracer HLSL (complex shader) triggers register spilling.
; This is the LLVM IR output from compiling pathtracer.hlsl via clang HLSL frontend.
;
; CHECK: mul{{.*}}r30
; CHECK: add{{.*}}r30{{.*}}0x380000
; CHECK: st_scatter{{.*}}r30
; CHECK: ld_scatter{{.*}}r30

; ModuleID = '/home/selimsandal/Developer/gpu/Source/Host/pathtracer/pathtracer.hlsl'
source_filename = "/home/selimsandal/Developer/gpu/Source/Host/pathtracer/pathtracer.hlsl"
target datalayout = "e-m:e-ve-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.0-pc-shadermodel6.0-compute"

@.str = private unnamed_addr constant [5 x i8] c"g_fb\00", align 1
@.str.2 = private unnamed_addr constant [8 x i8] c"g_accum\00", align 1
@.str.4 = private unnamed_addr constant [9 x i8] c"g_params\00", align 1

; Function Attrs: convergent nofree noinline norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none)
define void @CSMain() local_unnamed_addr #0 {
  %1 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %2 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.2)
  %3 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str.4)
  %4 = tail call i32 @llvm.dx.thread.id(i32 0)
  %5 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %3, i32 0)
  %6 = load i32, ptr %5, align 4, !tbaa !3
  %7 = add i32 %6, 1
  %8 = sitofp i32 %7 to float
  %9 = fdiv reassoc nnan ninf nsz arcp afn float 1.000000e+00, %8
  %10 = icmp ult i32 %4, 64000
  br i1 %10, label %11, label %395

11:                                               ; preds = %0
  %12 = mul i32 %6, -1640531535
  %13 = add i32 %12, 12345
  br label %14

14:                                               ; preds = %11, %379
  %15 = phi i32 [ %4, %11 ], [ %393, %379 ]
  %16 = uitofp nneg i32 %15 to float
  %17 = fmul reassoc nnan ninf nsz arcp afn float %16, 0x3F699999A0000000
  %18 = fptoui float %17 to i32
  %19 = mul i32 %18, -320
  %20 = add i32 %19, %15
  %21 = mul i32 %15, 1099087573
  %22 = add i32 %13, %21
  %23 = shl i32 %22, 13
  %24 = xor i32 %23, %22
  %25 = lshr i32 %24, 17
  %26 = xor i32 %25, %24
  %27 = shl i32 %26, 5
  %28 = xor i32 %27, %26
  %29 = lshr i32 %28, 9
  %30 = or disjoint i32 %29, 1065353216
  %31 = bitcast i32 %30 to float
  %32 = shl i32 %28, 13
  %33 = xor i32 %32, %28
  %34 = lshr i32 %33, 17
  %35 = xor i32 %34, %33
  %36 = shl i32 %35, 5
  %37 = xor i32 %36, %35
  %38 = lshr i32 %37, 9
  %39 = or disjoint i32 %38, 1065353216
  %40 = bitcast i32 %39 to float
  %41 = sitofp i32 %20 to float
  %42 = fadd reassoc nnan ninf nsz arcp afn float %41, -1.000000e+00
  %43 = fadd reassoc nnan ninf nsz arcp afn float %42, %31
  %44 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %43, float 0x3F799999A0000000, float -1.000000e+00)
  %45 = sitofp i32 %18 to float
  %46 = fadd reassoc nnan ninf nsz arcp afn float %45, -1.000000e+00
  %47 = fadd reassoc nnan ninf nsz arcp afn float %46, %40
  %48 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %47, float 0x3F847AE140000000, float -1.000000e+00)
  %49 = fneg reassoc nnan ninf nsz arcp afn float %48
  %50 = fmul reassoc nnan ninf nsz arcp afn float %48, %48
  %51 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %44, float %44, float %50)
  %52 = fadd reassoc nnan ninf nsz arcp afn float %51, 0x4007FFA3C0000000
  %53 = bitcast float %52 to i32
  %54 = lshr i32 %53, 1
  %55 = sub nuw nsw i32 1597463007, %54
  %56 = bitcast i32 %55 to float
  %57 = fmul reassoc nnan ninf nsz arcp afn float %52, -5.000000e-01
  %58 = fmul reassoc nnan ninf nsz arcp afn float %57, %56
  %59 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %58, float %56, float 1.500000e+00)
  %60 = fmul reassoc nnan ninf nsz arcp afn float %59, %56
  %61 = fmul reassoc nnan ninf nsz arcp afn float %60, %44
  %62 = fmul reassoc nnan ninf nsz arcp afn float %60, %49
  %63 = fmul reassoc nnan ninf nsz arcp afn float %60, 0xBFFBB645A0000000
  br label %64

64:                                               ; preds = %14, %314
  %65 = phi i32 [ %37, %14 ], [ %259, %314 ]
  %66 = phi float [ 0.000000e+00, %14 ], [ %276, %314 ]
  %67 = phi float [ 5.000000e-01, %14 ], [ %279, %314 ]
  %68 = phi float [ 3.000000e+00, %14 ], [ %282, %314 ]
  %69 = phi float [ %61, %14 ], [ %317, %314 ]
  %70 = phi float [ %62, %14 ], [ %316, %314 ]
  %71 = phi float [ %63, %14 ], [ %315, %314 ]
  %72 = phi float [ 1.000000e+00, %14 ], [ %283, %314 ]
  %73 = phi i32 [ 0, %14 ], [ %318, %314 ]
  %74 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fabs.f32(float %70)
  %75 = fcmp reassoc nnan ninf nsz arcp afn ogt float %74, 0x3F1A36E2E0000000
  br i1 %75, label %76, label %93

76:                                               ; preds = %64
  %77 = fsub reassoc nnan ninf nsz arcp afn float -5.000000e-01, %67
  %78 = fdiv reassoc nnan ninf nsz arcp afn float %77, %70
  %79 = fcmp reassoc nnan ninf nsz arcp afn ogt float %78, 0x3F60624DE0000000
  %80 = fcmp reassoc nnan ninf nsz arcp afn olt float %78, 1.000000e+02
  %81 = select i1 %79, i1 %80, i1 false
  br i1 %81, label %82, label %93

82:                                               ; preds = %76
  %83 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %78, float %69, float %66)
  %84 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %78, float %71, float %68)
  %85 = fadd reassoc nnan ninf nsz arcp afn float %83, 1.000050e+04
  %86 = fptosi float %85 to i32
  %87 = fadd reassoc nnan ninf nsz arcp afn float %84, 1.000050e+04
  %88 = fptosi float %87 to i32
  %89 = add nsw i32 %86, %88
  %90 = and i32 %89, 1
  %91 = uitofp nneg i32 %90 to float
  %92 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %91, float 5.000000e-01, float 0x3FD3333340000000)
  br label %93

93:                                               ; preds = %82, %76, %64
  %94 = phi i32 [ 0, %64 ], [ 1, %82 ], [ 0, %76 ]
  %95 = phi nsz float [ 5.000000e-01, %64 ], [ %92, %82 ], [ 5.000000e-01, %76 ]
  %96 = phi nsz float [ 1.000000e+02, %64 ], [ %78, %82 ], [ 1.000000e+02, %76 ]
  %97 = fmul reassoc nnan ninf nsz arcp afn float %67, %70
  %98 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %66, float %69, float %97)
  %99 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %68, float %71, float %98)
  %100 = fmul reassoc nnan ninf nsz arcp afn float %67, %67
  %101 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %66, float %66, float %100)
  %102 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %68, float %68, float %101)
  %103 = fsub reassoc nnan ninf nsz arcp afn float 2.500000e-01, %102
  %104 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %99, float %99, float %103)
  %105 = fcmp reassoc nnan ninf nsz arcp afn ogt float %104, 0.000000e+00
  br i1 %105, label %106, label %128

106:                                              ; preds = %93
  %107 = bitcast float %104 to i32
  %108 = lshr i32 %107, 1
  %109 = sub nuw nsw i32 1597463007, %108
  %110 = bitcast i32 %109 to float
  %111 = fmul reassoc nnan ninf nsz arcp afn float %104, -5.000000e-01
  %112 = fmul reassoc nnan ninf nsz arcp afn float %111, %110
  %113 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %112, float %110, float 1.500000e+00)
  %114 = fmul reassoc nnan ninf nsz arcp afn float %104, %110
  %115 = fmul reassoc nnan ninf nsz arcp afn float %114, %113
  %116 = fadd reassoc nnan ninf nsz arcp afn float %115, %99
  %117 = fneg reassoc nnan ninf nsz arcp afn float %116
  %118 = fcmp reassoc nnan ninf nsz arcp afn olt float %116, 0xBF60624DE0000000
  %119 = fcmp reassoc nnan ninf nsz arcp afn ogt float %96, %117
  %120 = select i1 %118, i1 %119, i1 false
  br i1 %120, label %121, label %128

121:                                              ; preds = %106
  %122 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %117, float %69, float %66)
  %123 = fmul reassoc nnan ninf nsz arcp afn float %122, 2.000000e+00
  %124 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %117, float %70, float %67)
  %125 = fmul reassoc nnan ninf nsz arcp afn float %124, 2.000000e+00
  %126 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %117, float %71, float %68)
  %127 = fmul reassoc nnan ninf nsz arcp afn float %126, 2.000000e+00
  br label %128

128:                                              ; preds = %121, %106, %93
  %129 = phi i32 [ %94, %93 ], [ 2, %121 ], [ %94, %106 ]
  %130 = phi nsz float [ %95, %93 ], [ 0x3FE6666660000000, %121 ], [ %95, %106 ]
  %131 = phi nsz float [ 0.000000e+00, %93 ], [ %127, %121 ], [ 0.000000e+00, %106 ]
  %132 = phi nsz float [ 1.000000e+00, %93 ], [ %125, %121 ], [ 1.000000e+00, %106 ]
  %133 = phi nsz float [ 0.000000e+00, %93 ], [ %123, %121 ], [ 0.000000e+00, %106 ]
  %134 = phi nsz float [ %96, %93 ], [ %117, %121 ], [ %96, %106 ]
  %135 = fadd reassoc nnan ninf nsz arcp afn float %66, 0xBFF19999A0000000
  %136 = fadd reassoc nnan ninf nsz arcp afn float %67, 0x3FC3333340000000
  %137 = fadd reassoc nnan ninf nsz arcp afn float %68, -5.000000e-01
  %138 = fmul reassoc nnan ninf nsz arcp afn float %136, %70
  %139 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %135, float %69, float %138)
  %140 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %137, float %71, float %139)
  %141 = fmul reassoc nnan ninf nsz arcp afn float %136, %136
  %142 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %135, float %135, float %141)
  %143 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %137, float %137, float %142)
  %144 = fsub reassoc nnan ninf nsz arcp afn float 0x3FBF5C2900000000, %143
  %145 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %140, float %140, float %144)
  %146 = fcmp reassoc nnan ninf nsz arcp afn ogt float %145, 0.000000e+00
  br i1 %146, label %147, label %172

147:                                              ; preds = %128
  %148 = bitcast float %145 to i32
  %149 = lshr i32 %148, 1
  %150 = sub nuw nsw i32 1597463007, %149
  %151 = bitcast i32 %150 to float
  %152 = fmul reassoc nnan ninf nsz arcp afn float %145, -5.000000e-01
  %153 = fmul reassoc nnan ninf nsz arcp afn float %152, %151
  %154 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %153, float %151, float 1.500000e+00)
  %155 = fmul reassoc nnan ninf nsz arcp afn float %145, %151
  %156 = fmul reassoc nnan ninf nsz arcp afn float %155, %154
  %157 = fadd reassoc nnan ninf nsz arcp afn float %156, %140
  %158 = fneg reassoc nnan ninf nsz arcp afn float %157
  %159 = fcmp reassoc nnan ninf nsz arcp afn olt float %157, 0xBF60624DE0000000
  %160 = fcmp reassoc nnan ninf nsz arcp afn ogt float %134, %158
  %161 = select i1 %159, i1 %160, i1 false
  br i1 %161, label %162, label %172

162:                                              ; preds = %147
  %163 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %158, float %69, float %66)
  %164 = fmul reassoc nnan ninf nsz arcp afn float %163, 0x4006DB6DC0000000
  %165 = fadd reassoc nnan ninf nsz arcp afn float %164, 0xC009249260000000
  %166 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %158, float %70, float %67)
  %167 = fmul reassoc nnan ninf nsz arcp afn float %166, 0x4006DB6DC0000000
  %168 = fadd reassoc nnan ninf nsz arcp afn float %167, 0x3FDB6DB700000000
  %169 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %158, float %71, float %68)
  %170 = fmul reassoc nnan ninf nsz arcp afn float %169, 0x4006DB6DC0000000
  %171 = fadd reassoc nnan ninf nsz arcp afn float %170, 0xBFF6DB6DC0000000
  br label %172

172:                                              ; preds = %162, %147, %128
  %173 = phi i32 [ %129, %128 ], [ 3, %162 ], [ %129, %147 ]
  %174 = phi nsz float [ %130, %128 ], [ 0x3FE3333340000000, %162 ], [ %130, %147 ]
  %175 = phi nsz float [ %131, %128 ], [ %171, %162 ], [ %131, %147 ]
  %176 = phi nsz float [ %132, %128 ], [ %168, %162 ], [ %132, %147 ]
  %177 = phi nsz float [ %133, %128 ], [ %165, %162 ], [ %133, %147 ]
  %178 = phi nsz float [ %134, %128 ], [ %158, %162 ], [ %134, %147 ]
  %179 = fadd reassoc nnan ninf nsz arcp afn float %66, 0x3FECCCCCC0000000
  %180 = fadd reassoc nnan ninf nsz arcp afn float %67, 0x3FB99999A0000000
  %181 = fadd reassoc nnan ninf nsz arcp afn float %68, 0x3FC99999A0000000
  %182 = fmul reassoc nnan ninf nsz arcp afn float %180, %70
  %183 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %179, float %69, float %182)
  %184 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %181, float %71, float %183)
  %185 = fmul reassoc nnan ninf nsz arcp afn float %180, %180
  %186 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %179, float %179, float %185)
  %187 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %181, float %181, float %186)
  %188 = fsub reassoc nnan ninf nsz arcp afn float 0x3FC47AE140000000, %187
  %189 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %184, float %184, float %188)
  %190 = fcmp reassoc nnan ninf nsz arcp afn ogt float %189, 0.000000e+00
  br i1 %190, label %191, label %216

191:                                              ; preds = %172
  %192 = bitcast float %189 to i32
  %193 = lshr i32 %192, 1
  %194 = sub nuw nsw i32 1597463007, %193
  %195 = bitcast i32 %194 to float
  %196 = fmul reassoc nnan ninf nsz arcp afn float %189, -5.000000e-01
  %197 = fmul reassoc nnan ninf nsz arcp afn float %196, %195
  %198 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %197, float %195, float 1.500000e+00)
  %199 = fmul reassoc nnan ninf nsz arcp afn float %189, %195
  %200 = fmul reassoc nnan ninf nsz arcp afn float %199, %198
  %201 = fadd reassoc nnan ninf nsz arcp afn float %200, %184
  %202 = fneg reassoc nnan ninf nsz arcp afn float %201
  %203 = fcmp reassoc nnan ninf nsz arcp afn olt float %201, 0xBF60624DE0000000
  %204 = fcmp reassoc nnan ninf nsz arcp afn ogt float %178, %202
  %205 = select i1 %203, i1 %204, i1 false
  br i1 %205, label %206, label %216

206:                                              ; preds = %191
  %207 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %202, float %69, float %66)
  %208 = fmul reassoc nnan ninf nsz arcp afn float %207, 2.500000e+00
  %209 = fadd reassoc nnan ninf nsz arcp afn float %208, 2.250000e+00
  %210 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %202, float %70, float %67)
  %211 = fmul reassoc nnan ninf nsz arcp afn float %210, 2.500000e+00
  %212 = fadd reassoc nnan ninf nsz arcp afn float %211, 2.500000e-01
  %213 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %202, float %71, float %68)
  %214 = fmul reassoc nnan ninf nsz arcp afn float %213, 2.500000e+00
  %215 = fadd reassoc nnan ninf nsz arcp afn float %214, 5.000000e-01
  br label %225

216:                                              ; preds = %191, %172
  %217 = icmp eq i32 %173, 0
  br i1 %217, label %218, label %225

218:                                              ; preds = %216
  %219 = fmul reassoc nnan ninf nsz arcp afn float %70, 5.000000e-01
  %220 = fsub reassoc nnan ninf nsz arcp afn float -5.000000e-01, %219
  %221 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %220, float 5.000000e-01, float 1.000000e+00)
  %222 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %220, float 0x3FD3333340000000, float 1.000000e+00)
  %223 = fmul reassoc nnan ninf nsz arcp afn float %221, %72
  %224 = fmul reassoc nnan ninf nsz arcp afn float %222, %72
  br label %320

225:                                              ; preds = %206, %216
  %226 = phi float [ %202, %206 ], [ %178, %216 ]
  %227 = phi float [ %209, %206 ], [ %177, %216 ]
  %228 = phi float [ %212, %206 ], [ %176, %216 ]
  %229 = phi float [ %215, %206 ], [ %175, %216 ]
  %230 = phi float [ 0x3FE4CCCCC0000000, %206 ], [ %174, %216 ]
  br label %231

231:                                              ; preds = %231, %225
  %232 = phi i32 [ %65, %225 ], [ %259, %231 ]
  %233 = phi i32 [ 0, %225 ], [ %270, %231 ]
  %234 = shl i32 %232, 13
  %235 = xor i32 %234, %232
  %236 = lshr i32 %235, 17
  %237 = xor i32 %236, %235
  %238 = shl i32 %237, 5
  %239 = xor i32 %238, %237
  %240 = lshr i32 %239, 9
  %241 = or disjoint i32 %240, 1065353216
  %242 = bitcast i32 %241 to float
  %243 = fadd reassoc nnan ninf nsz arcp afn float %242, -1.500000e+00
  %244 = shl i32 %239, 13
  %245 = xor i32 %244, %239
  %246 = lshr i32 %245, 17
  %247 = xor i32 %246, %245
  %248 = shl i32 %247, 5
  %249 = xor i32 %248, %247
  %250 = lshr i32 %249, 9
  %251 = or disjoint i32 %250, 1065353216
  %252 = bitcast i32 %251 to float
  %253 = fadd reassoc nnan ninf nsz arcp afn float %252, -1.500000e+00
  %254 = shl i32 %249, 13
  %255 = xor i32 %254, %249
  %256 = lshr i32 %255, 17
  %257 = xor i32 %256, %255
  %258 = shl i32 %257, 5
  %259 = xor i32 %258, %257
  %260 = lshr i32 %259, 9
  %261 = or disjoint i32 %260, 1065353216
  %262 = bitcast i32 %261 to float
  %263 = fadd reassoc nnan ninf nsz arcp afn float %262, -1.500000e+00
  %264 = fmul reassoc nnan ninf nsz arcp afn float %253, %253
  %265 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %243, float %243, float %264)
  %266 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %263, float %263, float %265)
  %267 = fcmp reassoc nnan ninf nsz arcp afn uge float %266, 2.500000e-01
  %268 = fcmp reassoc nnan ninf nsz arcp afn ule float %266, 0x3F1A36E2E0000000
  %269 = select i1 %267, i1 true, i1 %268
  %270 = add nuw nsw i32 %233, 1
  %271 = icmp samesign ult i32 %233, 15
  %272 = select i1 %269, i1 %271, i1 false
  br i1 %272, label %231, label %273, !llvm.loop !7

273:                                              ; preds = %231
  %274 = fmul reassoc nnan ninf nsz arcp afn float %227, 0x3F60624DE0000000
  %275 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %226, float %69, float %274)
  %276 = fadd reassoc nnan ninf nsz arcp afn float %275, %66
  %277 = fmul reassoc nnan ninf nsz arcp afn float %228, 0x3F60624DE0000000
  %278 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %226, float %70, float %277)
  %279 = fadd reassoc nnan ninf nsz arcp afn float %278, %67
  %280 = fmul reassoc nnan ninf nsz arcp afn float %229, 0x3F60624DE0000000
  %281 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %226, float %71, float %280)
  %282 = fadd reassoc nnan ninf nsz arcp afn float %281, %68
  %283 = fmul reassoc nnan ninf nsz arcp afn float %230, %72
  %284 = bitcast float %266 to i32
  %285 = lshr i32 %284, 1
  %286 = sub nsw i32 1597463007, %285
  %287 = bitcast i32 %286 to float
  %288 = fmul reassoc nnan ninf nsz arcp afn float %266, -5.000000e-01
  %289 = fmul reassoc nnan ninf nsz arcp afn float %288, %287
  %290 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %289, float %287, float 1.500000e+00)
  %291 = fmul reassoc nnan ninf nsz arcp afn float %290, %287
  %292 = fmul reassoc nnan ninf nsz arcp afn float %291, %243
  %293 = fmul reassoc nnan ninf nsz arcp afn float %291, %253
  %294 = fmul reassoc nnan ninf nsz arcp afn float %291, %263
  %295 = fadd reassoc nnan ninf nsz arcp afn float %292, %227
  %296 = fadd reassoc nnan ninf nsz arcp afn float %293, %228
  %297 = fadd reassoc nnan ninf nsz arcp afn float %294, %229
  %298 = fmul reassoc nnan ninf nsz arcp afn float %296, %296
  %299 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %295, float %295, float %298)
  %300 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %297, float %297, float %299)
  %301 = fcmp reassoc nnan ninf nsz arcp afn ogt float %300, 0x3F1A36E2E0000000
  br i1 %301, label %302, label %314

302:                                              ; preds = %273
  %303 = bitcast float %300 to i32
  %304 = lshr i32 %303, 1
  %305 = sub nuw nsw i32 1597463007, %304
  %306 = bitcast i32 %305 to float
  %307 = fmul reassoc nnan ninf nsz arcp afn float %300, -5.000000e-01
  %308 = fmul reassoc nnan ninf nsz arcp afn float %307, %306
  %309 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %308, float %306, float 1.500000e+00)
  %310 = fmul reassoc nnan ninf nsz arcp afn float %309, %306
  %311 = fmul reassoc nnan ninf nsz arcp afn float %310, %295
  %312 = fmul reassoc nnan ninf nsz arcp afn float %310, %296
  %313 = fmul reassoc nnan ninf nsz arcp afn float %310, %297
  br label %314

314:                                              ; preds = %273, %302
  %315 = phi nsz float [ %313, %302 ], [ %229, %273 ]
  %316 = phi nsz float [ %312, %302 ], [ %228, %273 ]
  %317 = phi nsz float [ %311, %302 ], [ %227, %273 ]
  %318 = add nuw nsw i32 %73, 1
  %319 = icmp eq i32 %318, 3
  br i1 %319, label %320, label %64, !llvm.loop !9

320:                                              ; preds = %314, %218
  %321 = phi nsz float [ %72, %218 ], [ 0.000000e+00, %314 ]
  %322 = phi nsz float [ %224, %218 ], [ 0.000000e+00, %314 ]
  %323 = phi nsz float [ %223, %218 ], [ 0.000000e+00, %314 ]
  %324 = mul nuw nsw i32 %15, 3
  %325 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %2, i32 %324)
  %326 = load float, ptr %325, align 4, !tbaa !3
  %327 = add nuw nsw i32 %324, 1
  %328 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %2, i32 %327)
  %329 = load float, ptr %328, align 4, !tbaa !3
  %330 = add nuw nsw i32 %324, 2
  %331 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %2, i32 %330)
  %332 = load float, ptr %331, align 4, !tbaa !3
  %333 = fadd reassoc nnan ninf nsz arcp afn float %326, %323
  %334 = fadd reassoc nnan ninf nsz arcp afn float %329, %322
  %335 = fadd reassoc nnan ninf nsz arcp afn float %332, %321
  store float %333, ptr %325, align 4, !tbaa !3
  store float %334, ptr %328, align 4, !tbaa !3
  store float %335, ptr %331, align 4, !tbaa !3
  %336 = fmul reassoc nnan ninf nsz arcp afn float %333, %9
  %337 = tail call reassoc nnan ninf nsz arcp afn float @llvm.dx.saturate.f32(float %336)
  %338 = fmul reassoc nnan ninf nsz arcp afn float %334, %9
  %339 = tail call reassoc nnan ninf nsz arcp afn float @llvm.dx.saturate.f32(float %338)
  %340 = fmul reassoc nnan ninf nsz arcp afn float %335, %9
  %341 = tail call reassoc nnan ninf nsz arcp afn float @llvm.dx.saturate.f32(float %340)
  %342 = fcmp reassoc nnan ninf nsz arcp afn ogt float %337, 0x3F50624DE0000000
  br i1 %342, label %343, label %353

343:                                              ; preds = %320
  %344 = bitcast float %337 to i32
  %345 = lshr i32 %344, 1
  %346 = sub nuw nsw i32 1597463007, %345
  %347 = bitcast i32 %346 to float
  %348 = fmul reassoc nnan ninf nsz arcp afn float %337, -5.000000e-01
  %349 = fmul reassoc nnan ninf nsz arcp afn float %348, %347
  %350 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %349, float %347, float 1.500000e+00)
  %351 = fmul reassoc nnan ninf nsz arcp afn float %337, %347
  %352 = fmul reassoc nnan ninf nsz arcp afn float %351, %350
  br label %353

353:                                              ; preds = %343, %320
  %354 = phi nsz float [ %352, %343 ], [ %337, %320 ]
  %355 = fcmp reassoc nnan ninf nsz arcp afn ogt float %339, 0x3F50624DE0000000
  br i1 %355, label %356, label %366

356:                                              ; preds = %353
  %357 = bitcast float %339 to i32
  %358 = lshr i32 %357, 1
  %359 = sub nuw nsw i32 1597463007, %358
  %360 = bitcast i32 %359 to float
  %361 = fmul reassoc nnan ninf nsz arcp afn float %339, -5.000000e-01
  %362 = fmul reassoc nnan ninf nsz arcp afn float %361, %360
  %363 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %362, float %360, float 1.500000e+00)
  %364 = fmul reassoc nnan ninf nsz arcp afn float %339, %360
  %365 = fmul reassoc nnan ninf nsz arcp afn float %364, %363
  br label %366

366:                                              ; preds = %356, %353
  %367 = phi nsz float [ %365, %356 ], [ %339, %353 ]
  %368 = fcmp reassoc nnan ninf nsz arcp afn ogt float %341, 0x3F50624DE0000000
  br i1 %368, label %369, label %379

369:                                              ; preds = %366
  %370 = bitcast float %341 to i32
  %371 = lshr i32 %370, 1
  %372 = sub nuw nsw i32 1597463007, %371
  %373 = bitcast i32 %372 to float
  %374 = fmul reassoc nnan ninf nsz arcp afn float %341, -5.000000e-01
  %375 = fmul reassoc nnan ninf nsz arcp afn float %374, %373
  %376 = tail call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float %375, float %373, float 1.500000e+00)
  %377 = fmul reassoc nnan ninf nsz arcp afn float %341, %373
  %378 = fmul reassoc nnan ninf nsz arcp afn float %377, %376
  br label %379

379:                                              ; preds = %369, %366
  %380 = phi nsz float [ %378, %369 ], [ %341, %366 ]
  %381 = fmul reassoc nnan ninf nsz arcp afn float %354, 2.550000e+02
  %382 = fptosi float %381 to i32
  %383 = fmul reassoc nnan ninf nsz arcp afn float %367, 2.550000e+02
  %384 = fptosi float %383 to i32
  %385 = fmul reassoc nnan ninf nsz arcp afn float %380, 2.550000e+02
  %386 = fptosi float %385 to i32
  %387 = shl i32 %382, 16
  %388 = shl i32 %384, 8
  %389 = or i32 %387, %388
  %390 = or i32 %389, %386
  %391 = or i32 %390, -16777216
  %392 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %1, i32 %15)
  store i32 %391, ptr %392, align 4, !tbaa !3
  %393 = add nuw nsw i32 %15, 32
  %394 = icmp ult i32 %15, 63968
  br i1 %394, label %14, label %395, !llvm.loop !10

395:                                              ; preds = %379, %0
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.dx.thread.id(i32) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.dx.saturate.f32(float) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32, i32, i32, i32, ptr) #3

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0), i32) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #5

attributes #0 = { convergent nofree noinline norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none) "frame-pointer"="all" "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #5 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!dx.valver = !{!0}
!llvm.module.flags = !{!1}
!llvm.ident = !{!2}
!llvm.errno.tbaa = !{!3}

!0 = !{i32 1, i32 8}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 23.0.0git (https://github.com/selimsandal/llvm-project f7442a79e35f618913970d3f88b81ab5488a68a4)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.mustprogress"}
!9 = distinct !{!9, !8}
!10 = distinct !{!10, !8}
