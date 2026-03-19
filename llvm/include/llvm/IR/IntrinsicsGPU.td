// GPU target-specific intrinsics for cross-lane REDUCE operations.
// The SIMT model presents scalar code to the compiler; each lane appears
// as a separate thread. REDUCE is a hardware cross-lane operation that
// reduces across all active lanes and broadcasts the result.

let TargetPrefix = "gpu" in {

def int_gpu_getsr : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem]>;

def int_gpu_workgroup_sync : DefaultAttrsIntrinsic<[], [llvm_i32_ty],
                                   [IntrWriteMem, IntrInaccessibleMemOnly,
                                    IntrConvergent]>;

def int_gpu_mem_fence : DefaultAttrsIntrinsic<[], [llvm_i32_ty],
                            [IntrWriteMem, IntrInaccessibleMemOnly,
                             IntrConvergent]>;

// Integer reductions (i32 -> i32)
def int_gpu_reduce_add  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrConvergent]>;
def int_gpu_reduce_and  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrConvergent]>;
def int_gpu_reduce_or   : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrConvergent]>;
def int_gpu_reduce_xor  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrConvergent]>;
def int_gpu_reduce_smin : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrConvergent]>;
def int_gpu_reduce_smax : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrConvergent]>;
def int_gpu_reduce_umin : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrConvergent]>;
def int_gpu_reduce_umax : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrConvergent]>;

// Float reductions (f32 -> f32)
def int_gpu_reduce_fadd : DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty],
                            [IntrNoMem, IntrConvergent]>;
def int_gpu_reduce_fmin : DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty],
                            [IntrNoMem, IntrConvergent]>;
def int_gpu_reduce_fmax : DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty],
                            [IntrNoMem, IntrConvergent]>;

} // TargetPrefix = "gpu"
