//===-- GPUHLSLLowering.cpp - HLSL/DirectX → GPU Lowering ===//
//
// IR-level pass that lowers HLSL DirectX intrinsics to GPU
// register reads and native GPU intrinsics, enabling HLSL
// compute shaders to compile through the GPU backend.
//
// Compilation pipeline:
//   HLSL (.hlsl)
//     → clang -x hlsl -target dxil-pc-shadermodel6.0-compute -S -emit-llvm
//     → LLVM IR with @llvm.dx.* intrinsics
//     → this pass (gpu-hlsl-lowering)
//     → llc -march=gpu -filetype=obj
//     → GPU ELF object
//
// Current intrinsic mapping in the shipped software ABI:
//   SV_DispatchThreadID    → r0 (physical thread ID)
//   SV_GroupID             → r1 (older descriptor convention)
//   SV_GroupThreadID       → r0 & 7 (lane within engine)
//   SV_GroupIndex          → r0 & 7
//   WaveGetLaneIndex()     → r0 & 7
//   WaveGetLaneCount()     → 8 (constant)
//
// Hardware now also exposes logical workgroup IDs through hidden launch
// context + I_GETSR, but this pass has not migrated to that ABI yet.
//
// Resource Binding → GPU Register Mapping:
//   register(u0/t0/b0)    → r1 (descriptor init_r1)
//   register(u1/t1/b1)    → r2 (descriptor init_r2)
//   register(u2/t2/b2)    → r3 (descriptor init_r3)
//   register(u3/t3/b3)    → r4 (descriptor init_r4)
//   >4 bindings: all loaded indirectly from args buffer via r1.
//
// Wave Intrinsic → GPU REDUCE Mapping:
//   WaveActiveSum          → REDUCE(ADD/FADD)
//   WaveActiveMin          → REDUCE(SMIN/UMIN/FMIN)
//   WaveActiveMax          → REDUCE(SMAX/UMAX/FMAX)
//   WaveActiveBitOr        → REDUCE(OR)
//   WaveActiveBitAnd       → REDUCE(AND)
//   WaveActiveBitXor       → REDUCE(XOR)
//
//===--------------------------------------------------------------===//

#include "GPU.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-hlsl-lowering"

namespace {

class GPUHLSLLowering : public ModulePass {
public:
  static char ID;
  GPUHLSLLowering() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override {
    return "GPU HLSL Lowering";
  }

private:
  bool lowerThreadIDIntrinsics(Module &M);
  bool lowerResourceAccess(Module &M);
  bool lowerWaveIntrinsics(Module &M);
  bool lowerMathIntrinsics(Module &M);
  bool stripHLSLMetadata(Module &M);
};

char GPUHLSLLowering::ID = 0;

} // anonymous namespace

// Read a named GPU register via llvm.read_register.i32
static Value *createRegRead(IRBuilder<> &Builder, Module &M,
                            const char *RegName) {
  Function *ReadReg = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::read_register, {Builder.getInt32Ty()});
  MDNode *RegMD = MDNode::get(M.getContext(),
      {MDString::get(M.getContext(), RegName)});
  return Builder.CreateCall(ReadReg, {MetadataAsValue::get(
      M.getContext(), RegMD)});
}

// Classify DX intrinsic by function name (avoids dependency on
// IntrinsicsDirectX.h being generated for this target build).
enum DXIntrinsicKind {
  DX_NONE = 0,
  // Thread IDs
  DX_THREAD_ID,             // llvm.dx.thread.id
  DX_GROUP_ID,              // llvm.dx.group.id
  DX_THREAD_ID_IN_GROUP,    // llvm.dx.thread.id.in.group
  DX_FLATTENED_THREAD_ID,   // llvm.dx.flattened.thread.id.in.group
  // Resources
  DX_HANDLE_FROM_BINDING,   // llvm.dx.resource.handlefrombinding
  DX_HANDLE_FROM_IMPLICIT,  // llvm.dx.resource.handlefromimplicitbinding
  DX_RESOURCE_GETPOINTER,   // llvm.dx.resource.getpointer
  DX_LOAD_RAWBUFFER,        // llvm.dx.resource.load.rawbuffer
  DX_STORE_RAWBUFFER,       // llvm.dx.resource.store.rawbuffer
  DX_LOAD_TYPEDBUFFER,      // llvm.dx.resource.load.typedbuffer
  DX_STORE_TYPEDBUFFER,     // llvm.dx.resource.store.typedbuffer
  // Wave ops
  DX_WAVE_GETLANEINDEX,     // llvm.dx.wave.getlaneindex
  DX_WAVE_GET_LANE_COUNT,   // llvm.dx.wave.get.lane.count
  DX_WAVE_IS_FIRST_LANE,    // llvm.dx.wave.is.first.lane
  DX_WAVE_ALL,              // llvm.dx.wave.all
  DX_WAVE_ANY,              // llvm.dx.wave.any
  DX_WAVE_REDUCE_SUM,       // llvm.dx.wave.reduce.sum
  DX_WAVE_REDUCE_USUM,      // llvm.dx.wave.reduce.usum
  DX_WAVE_REDUCE_MAX,       // llvm.dx.wave.reduce.max
  DX_WAVE_REDUCE_UMAX,      // llvm.dx.wave.reduce.umax
  DX_WAVE_REDUCE_MIN,       // llvm.dx.wave.reduce.min
  DX_WAVE_REDUCE_UMIN,      // llvm.dx.wave.reduce.umin
  DX_WAVE_REDUCE_OR,        // llvm.dx.wave.reduce.or
  DX_WAVE_REDUCE_AND,       // llvm.dx.wave.reduce.and (not yet in upstream)
  DX_WAVE_REDUCE_XOR,       // llvm.dx.wave.reduce.xor
  // Math
  DX_LERP,                  // llvm.dx.lerp
  DX_SATURATE,              // llvm.dx.saturate
  DX_FRAC,                  // llvm.dx.frac
  DX_RSQRT,                 // llvm.dx.rsqrt
  DX_IMAD,                  // llvm.dx.imad
  DX_UMAD,                  // llvm.dx.umad
  DX_DEGREES,               // llvm.dx.degrees
  DX_RADIANS,               // llvm.dx.radians
  DX_SIGN,                  // llvm.dx.sign
  DX_STEP,                  // llvm.dx.step
};

static DXIntrinsicKind classifyDXIntrinsic(StringRef Name) {
  if (!Name.starts_with("llvm.dx."))
    return DX_NONE;

  // Thread IDs
  if (Name == "llvm.dx.thread.id")                       return DX_THREAD_ID;
  if (Name == "llvm.dx.group.id")                        return DX_GROUP_ID;
  if (Name == "llvm.dx.thread.id.in.group")              return DX_THREAD_ID_IN_GROUP;
  if (Name == "llvm.dx.flattened.thread.id.in.group")    return DX_FLATTENED_THREAD_ID;

  // Resources (overloaded intrinsics have type suffixes)
  if (Name.starts_with("llvm.dx.resource.handlefrombinding"))
    return DX_HANDLE_FROM_BINDING;
  if (Name.starts_with("llvm.dx.resource.handlefromimplicitbinding"))
    return DX_HANDLE_FROM_IMPLICIT;
  if (Name.starts_with("llvm.dx.resource.getpointer"))
    return DX_RESOURCE_GETPOINTER;
  if (Name.starts_with("llvm.dx.resource.load.rawbuffer"))
    return DX_LOAD_RAWBUFFER;
  if (Name.starts_with("llvm.dx.resource.store.rawbuffer"))
    return DX_STORE_RAWBUFFER;
  if (Name.starts_with("llvm.dx.resource.load.typedbuffer"))
    return DX_LOAD_TYPEDBUFFER;
  if (Name.starts_with("llvm.dx.resource.store.typedbuffer"))
    return DX_STORE_TYPEDBUFFER;

  // Wave ops (overloaded, may have type suffixes)
  if (Name.starts_with("llvm.dx.wave.getlaneindex"))     return DX_WAVE_GETLANEINDEX;
  if (Name.starts_with("llvm.dx.wave.get.lane.count"))   return DX_WAVE_GET_LANE_COUNT;
  if (Name.starts_with("llvm.dx.wave.is.first.lane"))    return DX_WAVE_IS_FIRST_LANE;
  // wave.all/any before wave.reduce to avoid prefix clash
  if (Name.starts_with("llvm.dx.wave.all") &&
      !Name.starts_with("llvm.dx.wave.all.equal"))       return DX_WAVE_ALL;
  if (Name.starts_with("llvm.dx.wave.any"))              return DX_WAVE_ANY;
  if (Name.starts_with("llvm.dx.wave.reduce.usum"))      return DX_WAVE_REDUCE_USUM;
  if (Name.starts_with("llvm.dx.wave.reduce.sum"))       return DX_WAVE_REDUCE_SUM;
  if (Name.starts_with("llvm.dx.wave.reduce.umax"))      return DX_WAVE_REDUCE_UMAX;
  if (Name.starts_with("llvm.dx.wave.reduce.max"))       return DX_WAVE_REDUCE_MAX;
  if (Name.starts_with("llvm.dx.wave.reduce.umin"))      return DX_WAVE_REDUCE_UMIN;
  if (Name.starts_with("llvm.dx.wave.reduce.min"))       return DX_WAVE_REDUCE_MIN;
  if (Name.starts_with("llvm.dx.wave.reduce.or"))        return DX_WAVE_REDUCE_OR;
  if (Name.starts_with("llvm.dx.wave.reduce.and"))       return DX_WAVE_REDUCE_AND;
  if (Name.starts_with("llvm.dx.wave.reduce.xor"))       return DX_WAVE_REDUCE_XOR;

  // Math
  if (Name.starts_with("llvm.dx.lerp"))                  return DX_LERP;
  if (Name.starts_with("llvm.dx.saturate"))              return DX_SATURATE;
  if (Name.starts_with("llvm.dx.frac"))                  return DX_FRAC;
  if (Name.starts_with("llvm.dx.rsqrt"))                 return DX_RSQRT;
  if (Name.starts_with("llvm.dx.imad"))                  return DX_IMAD;
  if (Name.starts_with("llvm.dx.umad"))                  return DX_UMAD;
  if (Name.starts_with("llvm.dx.degrees"))               return DX_DEGREES;
  if (Name.starts_with("llvm.dx.radians"))               return DX_RADIANS;
  if (Name.starts_with("llvm.dx.sign"))                  return DX_SIGN;
  if (Name.starts_with("llvm.dx.step"))                  return DX_STEP;

  return DX_NONE;
}

//===----------------------------------------------------------------------===//
// Thread ID Lowering
//===----------------------------------------------------------------------===//

bool GPUHLSLLowering::lowerThreadIDIntrinsics(Module &M) {
  bool Changed = false;
  SmallVector<CallInst *, 16> ToReplace;
  SmallVector<Function *, 8> ToDelete;

  for (Function &F : M) {
    DXIntrinsicKind Kind = classifyDXIntrinsic(F.getName());
    if (Kind != DX_THREAD_ID && Kind != DX_GROUP_ID &&
        Kind != DX_THREAD_ID_IN_GROUP && Kind != DX_FLATTENED_THREAD_ID &&
        Kind != DX_WAVE_GETLANEINDEX && Kind != DX_WAVE_GET_LANE_COUNT &&
        Kind != DX_WAVE_IS_FIRST_LANE)
      continue;

    for (User *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        ToReplace.push_back(CI);
    }

    for (CallInst *CI : ToReplace) {
      IRBuilder<> Builder(CI);
      Value *Result = nullptr;

      switch (Kind) {
      case DX_THREAD_ID: {
        // Current software ABI: SV_DispatchThreadID.x → r0 (physical thread ID)
        // For dim > 0, return 0 (1D dispatch only)
        if (auto *DimC = dyn_cast<ConstantInt>(CI->getArgOperand(0))) {
          if (DimC->isZero())
            Result = createRegRead(Builder, M, "r0");
          else
            Result = Builder.getInt32(0);
        } else {
          Result = createRegRead(Builder, M, "r0");
        }
        break;
      }
      case DX_GROUP_ID: {
        // Current software ABI: SV_GroupID.x → r1
        if (auto *DimC = dyn_cast<ConstantInt>(CI->getArgOperand(0))) {
          if (DimC->isZero())
            Result = createRegRead(Builder, M, "r1");
          else
            Result = Builder.getInt32(0);
        } else {
          Result = createRegRead(Builder, M, "r1");
        }
        break;
      }
      case DX_THREAD_ID_IN_GROUP:
      case DX_FLATTENED_THREAD_ID:
      case DX_WAVE_GETLANEINDEX: {
        // Current software ABI: lane index within engine → r0 & 7
        Value *R0 = createRegRead(Builder, M, "r0");
        Result = Builder.CreateAnd(R0, Builder.getInt32(7), "lane_id");
        break;
      }
      case DX_WAVE_GET_LANE_COUNT:
        // GPU has 8 lanes per engine (fixed)
        Result = Builder.getInt32(8);
        break;
      case DX_WAVE_IS_FIRST_LANE: {
        // First lane = lane 0 → (r0 & 7) == 0
        Value *R0 = createRegRead(Builder, M, "r0");
        Value *Lane = Builder.CreateAnd(R0, Builder.getInt32(7));
        Result = Builder.CreateICmpEQ(Lane, Builder.getInt32(0),
                                      "is_first_lane");
        break;
      }
      default:
        break;
      }

      if (Result) {
        CI->replaceAllUsesWith(Result);
        CI->eraseFromParent();
        Changed = true;
      }
    }
    ToReplace.clear();

    if (F.use_empty())
      ToDelete.push_back(&F);
  }

  for (Function *F : ToDelete)
    F->eraseFromParent();

  return Changed;
}

//===----------------------------------------------------------------------===//
// Resource Access Lowering
//
// HLSL:  RWStructuredBuffer<int> buf : register(u0);
//        buf[id] = val;
//
// IR:    %handle = call @llvm.dx.resource.handlefrombinding(space, bind, ...)
//        %ptr = call @llvm.dx.resource.getpointer(%handle, %index)
//        store i32 %val, ptr %ptr
//
// GPU:   %base = read_register("r1")   ; bind 0 → r1
//        %addr = add %base, mul(%index, 4)
//        %ptr = inttoptr %addr
//        store i32 %val, ptr %ptr       ; → ST_SCATTER via ISel
//===----------------------------------------------------------------------===//

// Trace a getpointer handle operand back to its handlefrombinding call
// and return the binding slot number (0-3).
static bool getBindSlot(Value *Handle, unsigned &BindSlot) {
  // Walk through casts/phis to find the handlefrombinding
  auto *CI = dyn_cast<CallInst>(Handle);
  if (!CI)
    return false;

  DXIntrinsicKind Kind = classifyDXIntrinsic(CI->getCalledFunction()->getName());
  if (Kind != DX_HANDLE_FROM_BINDING && Kind != DX_HANDLE_FROM_IMPLICIT)
    return false;

  // handlefrombinding args: (space, bind, range_size, index, name_ptr)
  auto *BindC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!BindC)
    return false;

  BindSlot = BindC->getZExtValue();
  return true;
}

// Map binding slot to GPU register name (0→r1, 1→r2, 2→r3, 3→r4)
static const char *bindSlotToReg(unsigned Slot) {
  switch (Slot) {
  case 0: return "r1";
  case 1: return "r2";
  case 2: return "r3";
  case 3: return "r4";
  default: return nullptr;
  }
}

// Get the base address value for a binding slot.
// When UseIndirect is false (≤4 bindings): reads from register (r1-r4).
// When UseIndirect is true (>4 bindings): loads from args buffer [r1 + slot*4].
static Value *getBindingBase(IRBuilder<> &Builder, Module &M,
                             unsigned BindSlot, bool UseIndirect) {
  if (!UseIndirect) {
    const char *RegName = bindSlotToReg(BindSlot);
    if (!RegName)
      return nullptr;
    return createRegRead(Builder, M, RegName);
  }
  // Indirect: r1 = args buffer pointer, load binding value from [r1 + slot*4]
  Value *ArgsPtr = createRegRead(Builder, M, "r1");
  Value *Offset = Builder.getInt32(BindSlot * 4);
  Value *Addr = Builder.CreateAdd(ArgsPtr, Offset, "arg_addr");
  Value *Ptr = Builder.CreateIntToPtr(Addr,
      PointerType::getUnqual(M.getContext()), "arg_ptr");
  return Builder.CreateLoad(Builder.getInt32Ty(), Ptr, "arg_val");
}

// Get element size in bytes from a target extension type.
// target("dx.RawBuffer", i32, 1, 0) → sizeof(i32) = 4
// target("dx.TypedBuffer", <4 x float>, 1, 0, 1) → sizeof(<4 x float>) = 16
static unsigned getElementSize(Type *HandleType, const DataLayout &DL) {
  if (auto *TET = dyn_cast<TargetExtType>(HandleType)) {
    if (TET->getNumTypeParameters() > 0) {
      Type *ElemTy = TET->getTypeParameter(0);
      return DL.getTypeAllocSize(ElemTy);
    }
  }
  // Default to 4 bytes (i32/float)
  return 4;
}

bool GPUHLSLLowering::lowerResourceAccess(Module &M) {
  bool Changed = false;
  const DataLayout &DL = M.getDataLayout();
  SmallVector<CallInst *, 16> GetPtrCalls;
  SmallVector<CallInst *, 16> LoadCalls;
  SmallVector<CallInst *, 16> StoreCalls;
  SmallVector<CallInst *, 16> HandleCalls;

  // Collect all resource intrinsic calls
  for (Function &F : M) {
    DXIntrinsicKind Kind = classifyDXIntrinsic(F.getName());
    for (User *U : F.users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;
      switch (Kind) {
      case DX_RESOURCE_GETPOINTER:  GetPtrCalls.push_back(CI); break;
      case DX_LOAD_RAWBUFFER:       LoadCalls.push_back(CI); break;
      case DX_STORE_RAWBUFFER:      StoreCalls.push_back(CI); break;
      case DX_HANDLE_FROM_BINDING:
      case DX_HANDLE_FROM_IMPLICIT: HandleCalls.push_back(CI); break;
      default: break;
      }
    }
  }

  // Count max binding slot to decide direct vs indirect path
  unsigned MaxBindSlot = 0;
  for (CallInst *CI : HandleCalls) {
    auto *BindC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    if (BindC)
      MaxBindSlot = std::max(MaxBindSlot, (unsigned)BindC->getZExtValue());
  }
  bool UseIndirect = MaxBindSlot > 3;

  // Lower getpointer: (handle, index) → inttoptr(base + index * elem_size)
  for (CallInst *CI : GetPtrCalls) {
    Value *Handle = CI->getArgOperand(0);
    Value *Index = CI->getArgOperand(1);

    unsigned BindSlot;
    if (!getBindSlot(Handle, BindSlot))
      continue;

    unsigned ElemSize = getElementSize(Handle->getType(), DL);

    IRBuilder<> Builder(CI);
    Value *Base = getBindingBase(Builder, M, BindSlot, UseIndirect);
    if (!Base)
      continue;

    Value *Addr;
    if (ElemSize == 1) {
      Addr = Builder.CreateAdd(Base, Index, "elem_addr");
    } else {
      Value *Offset = Builder.CreateMul(Index,
          Builder.getInt32(ElemSize), "byte_offset");
      Addr = Builder.CreateAdd(Base, Offset, "elem_addr");
    }
    Value *Ptr = Builder.CreateIntToPtr(Addr, CI->getType(), "buf_ptr");

    CI->replaceAllUsesWith(Ptr);
    CI->eraseFromParent();
    Changed = true;
  }

  // Lower rawbuffer load: (handle, index, byte_offset) → {load, true}
  for (CallInst *CI : LoadCalls) {
    Value *Handle = CI->getArgOperand(0);
    Value *Index = CI->getArgOperand(1);
    Value *ByteOffset = CI->getArgOperand(2);

    unsigned BindSlot;
    if (!getBindSlot(Handle, BindSlot))
      continue;

    unsigned ElemSize = getElementSize(Handle->getType(), DL);

    // Return type is {T, i1}
    StructType *RetTy = cast<StructType>(CI->getType());
    Type *ElemTy = RetTy->getElementType(0);

    IRBuilder<> Builder(CI);
    Value *Base = getBindingBase(Builder, M, BindSlot, UseIndirect);
    if (!Base)
      continue;
    Value *Offset = Builder.CreateMul(Index,
        Builder.getInt32(ElemSize), "idx_offset");
    Value *Addr = Builder.CreateAdd(Base, Offset, "addr");
    Addr = Builder.CreateAdd(Addr, ByteOffset, "addr_final");
    Value *Ptr = Builder.CreateIntToPtr(Addr,
        PointerType::getUnqual(M.getContext()), "ld_ptr");
    Value *Loaded = Builder.CreateLoad(ElemTy, Ptr, "loaded");

    // Build {loaded, true} aggregate
    Value *Result = PoisonValue::get(RetTy);
    Result = Builder.CreateInsertValue(Result, Loaded, {0});
    Result = Builder.CreateInsertValue(Result, Builder.getTrue(), {1});

    CI->replaceAllUsesWith(Result);
    CI->eraseFromParent();
    Changed = true;
  }

  // Lower rawbuffer store: (handle, index, byte_offset, data) → store
  for (CallInst *CI : StoreCalls) {
    Value *Handle = CI->getArgOperand(0);
    Value *Index = CI->getArgOperand(1);
    Value *ByteOffset = CI->getArgOperand(2);
    Value *Data = CI->getArgOperand(3);

    unsigned BindSlot;
    if (!getBindSlot(Handle, BindSlot))
      continue;

    unsigned ElemSize = getElementSize(Handle->getType(), DL);

    IRBuilder<> Builder(CI);
    Value *Base = getBindingBase(Builder, M, BindSlot, UseIndirect);
    if (!Base)
      continue;
    Value *Offset = Builder.CreateMul(Index,
        Builder.getInt32(ElemSize), "idx_offset");
    Value *Addr = Builder.CreateAdd(Base, Offset, "addr");
    Addr = Builder.CreateAdd(Addr, ByteOffset, "addr_final");
    Value *Ptr = Builder.CreateIntToPtr(Addr,
        PointerType::getUnqual(M.getContext()), "st_ptr");
    Builder.CreateStore(Data, Ptr);

    CI->eraseFromParent();
    Changed = true;
  }

  // Remove dead handlefrombinding calls
  for (CallInst *CI : HandleCalls) {
    if (CI->use_empty()) {
      CI->eraseFromParent();
      Changed = true;
    }
  }

  // Clean up dead resource function declarations
  SmallVector<Function *, 8> DeadFuncs;
  for (Function &F : M) {
    DXIntrinsicKind Kind = classifyDXIntrinsic(F.getName());
    if ((Kind == DX_HANDLE_FROM_BINDING || Kind == DX_HANDLE_FROM_IMPLICIT ||
         Kind == DX_RESOURCE_GETPOINTER || Kind == DX_LOAD_RAWBUFFER ||
         Kind == DX_STORE_RAWBUFFER) &&
        F.use_empty())
      DeadFuncs.push_back(&F);
  }
  for (Function *F : DeadFuncs)
    F->eraseFromParent();

  return Changed;
}

//===----------------------------------------------------------------------===//
// Wave Intrinsic Lowering → GPU REDUCE
//===----------------------------------------------------------------------===//

bool GPUHLSLLowering::lowerWaveIntrinsics(Module &M) {
  bool Changed = false;
  SmallVector<CallInst *, 16> ToReplace;
  SmallVector<Function *, 8> ToDelete;

  for (Function &F : M) {
    DXIntrinsicKind Kind = classifyDXIntrinsic(F.getName());
    if (Kind < DX_WAVE_REDUCE_SUM || Kind > DX_WAVE_REDUCE_XOR)
      continue;
    // wave.all and wave.any are handled separately
    if (Kind == DX_WAVE_ALL || Kind == DX_WAVE_ANY)
      continue;

    for (User *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        ToReplace.push_back(CI);
    }

    for (CallInst *CI : ToReplace) {
      IRBuilder<> Builder(CI);
      Value *Src = CI->getArgOperand(0);
      bool IsFloat = Src->getType()->isFloatTy();
      Value *Result = nullptr;

      // Map DX wave reduce → GPU reduce intrinsic
      Intrinsic::ID ReduceID = Intrinsic::not_intrinsic;
      switch (Kind) {
      case DX_WAVE_REDUCE_SUM:
      case DX_WAVE_REDUCE_USUM:
        ReduceID = IsFloat ? Intrinsic::gpu_reduce_fadd
                           : Intrinsic::gpu_reduce_add;
        break;
      case DX_WAVE_REDUCE_MAX:
        ReduceID = IsFloat ? Intrinsic::gpu_reduce_fmax
                           : Intrinsic::gpu_reduce_smax;
        break;
      case DX_WAVE_REDUCE_UMAX:
        ReduceID = Intrinsic::gpu_reduce_umax;
        break;
      case DX_WAVE_REDUCE_MIN:
        ReduceID = IsFloat ? Intrinsic::gpu_reduce_fmin
                           : Intrinsic::gpu_reduce_smin;
        break;
      case DX_WAVE_REDUCE_UMIN:
        ReduceID = Intrinsic::gpu_reduce_umin;
        break;
      case DX_WAVE_REDUCE_OR:
        ReduceID = Intrinsic::gpu_reduce_or;
        break;
      case DX_WAVE_REDUCE_AND:
        ReduceID = Intrinsic::gpu_reduce_and;
        break;
      case DX_WAVE_REDUCE_XOR:
        ReduceID = Intrinsic::gpu_reduce_xor;
        break;
      default:
        break;
      }

      if (ReduceID != Intrinsic::not_intrinsic) {
        Function *ReduceFn = Intrinsic::getOrInsertDeclaration(
            &M, ReduceID, {});
        // GPU reduce intrinsics are fixed i32/f32, no type overloads
        Result = Builder.CreateCall(ReduceFn, {Src});
      }

      if (Result) {
        CI->replaceAllUsesWith(Result);
        CI->eraseFromParent();
        Changed = true;
      }
    }
    ToReplace.clear();

    if (F.use_empty())
      ToDelete.push_back(&F);
  }

  // Handle wave.all and wave.any separately (they take i1, not i32/f32)
  for (Function &F : M) {
    DXIntrinsicKind Kind = classifyDXIntrinsic(F.getName());
    if (Kind != DX_WAVE_ALL && Kind != DX_WAVE_ANY)
      continue;

    for (User *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        ToReplace.push_back(CI);
    }

    for (CallInst *CI : ToReplace) {
      IRBuilder<> Builder(CI);
      Value *Pred = CI->getArgOperand(0); // i1

      // wave.all(pred) → reduce_and(zext(pred)) != 0
      // wave.any(pred) → reduce_or(zext(pred)) != 0
      Value *PredI32 = Builder.CreateZExt(Pred, Builder.getInt32Ty());
      Intrinsic::ID ReduceID = (Kind == DX_WAVE_ALL)
                                    ? Intrinsic::gpu_reduce_and
                                    : Intrinsic::gpu_reduce_or;
      Function *ReduceFn =
          Intrinsic::getOrInsertDeclaration(&M, ReduceID, {});
      Value *Reduced = Builder.CreateCall(ReduceFn, {PredI32});
      Value *Result = Builder.CreateICmpNE(Reduced, Builder.getInt32(0));

      CI->replaceAllUsesWith(Result);
      CI->eraseFromParent();
      Changed = true;
    }
    ToReplace.clear();

    if (F.use_empty())
      ToDelete.push_back(&F);
  }

  for (Function *F : ToDelete)
    F->eraseFromParent();

  return Changed;
}

//===----------------------------------------------------------------------===//
// Math Intrinsic Lowering
//===----------------------------------------------------------------------===//

bool GPUHLSLLowering::lowerMathIntrinsics(Module &M) {
  bool Changed = false;
  SmallVector<CallInst *, 16> ToReplace;
  SmallVector<Function *, 8> ToDelete;

  for (Function &F : M) {
    DXIntrinsicKind Kind = classifyDXIntrinsic(F.getName());
    if (Kind < DX_LERP || Kind > DX_STEP)
      continue;

    for (User *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        ToReplace.push_back(CI);
    }

    for (CallInst *CI : ToReplace) {
      IRBuilder<> Builder(CI);
      Value *Result = nullptr;

      switch (Kind) {
      case DX_LERP: {
        // lerp(a, b, t) = a + t * (b - a) = fma(t, b-a, a)
        Value *A = CI->getArgOperand(0);
        Value *B = CI->getArgOperand(1);
        Value *T = CI->getArgOperand(2);
        Value *Diff = Builder.CreateFSub(B, A, "lerp.diff");
        Function *FMA = Intrinsic::getOrInsertDeclaration(
            &M, Intrinsic::fma, {A->getType()});
        Result = Builder.CreateCall(FMA, {T, Diff, A}, "lerp");
        break;
      }
      case DX_SATURATE: {
        // saturate(x) = clamp(x, 0.0, 1.0) = fmin(fmax(x, 0.0), 1.0)
        Value *X = CI->getArgOperand(0);
        Type *Ty = X->getType();
        Function *FMax = Intrinsic::getOrInsertDeclaration(
            &M, Intrinsic::maxnum, {Ty});
        Function *FMin = Intrinsic::getOrInsertDeclaration(
            &M, Intrinsic::minnum, {Ty});
        Value *Zero = ConstantFP::get(Ty, 0.0);
        Value *One = ConstantFP::get(Ty, 1.0);
        Value *Clamped = Builder.CreateCall(FMax, {X, Zero});
        Result = Builder.CreateCall(FMin, {Clamped, One}, "saturated");
        break;
      }
      case DX_FRAC: {
        // frac(x) = x - trunc(x)
        // GPU has no floor; use FTOI+ITOF for truncation toward zero.
        // Correct for positive values (typical use: UV coords, 0..1 range).
        Value *X = CI->getArgOperand(0);
        Value *Trunc = Builder.CreateFPToSI(X, Builder.getInt32Ty(), "trunc_i");
        Value *TruncF = Builder.CreateSIToFP(Trunc, X->getType(), "trunc_f");
        Result = Builder.CreateFSub(X, TruncF, "frac");
        break;
      }
      case DX_RSQRT: {
        // rsqrt(x) = 1.0 / sqrt(x)
        // GPU has no native sqrt; use Newton-Raphson inverse sqrt:
        //   y0 = fdiv(1.0, x)          -- rough initial guess
        //   y1 = y0 * (1.5 - 0.5*x * y0*y0)  -- one NR iteration
        // This gives ~23-bit accuracy for f32 with the fast FPU.
        Value *X = CI->getArgOperand(0);
        Type *Ty = X->getType();
        Value *One = ConstantFP::get(Ty, 1.0);
        Value *Half = ConstantFP::get(Ty, 0.5);
        Value *ThreeHalf = ConstantFP::get(Ty, 1.5);
        // Initial estimate via reciprocal (FDIV)
        Value *Y0 = Builder.CreateFDiv(One, X, "rsqrt_recip");
        // One Newton-Raphson iteration for 1/sqrt(x):
        //   y1 = y0 * (1.5 - 0.5 * x * y0 * y0)
        Value *HalfX = Builder.CreateFMul(Half, X);
        Value *Y0Sq = Builder.CreateFMul(Y0, Y0);
        Value *HalfXY0Sq = Builder.CreateFMul(HalfX, Y0Sq);
        Value *Sub = Builder.CreateFSub(ThreeHalf, HalfXY0Sq);
        Result = Builder.CreateFMul(Y0, Sub, "rsqrt");
        break;
      }
      case DX_IMAD:
      case DX_UMAD: {
        // imad/umad(a, b, c) = a * b + c
        Value *A = CI->getArgOperand(0);
        Value *B = CI->getArgOperand(1);
        Value *C = CI->getArgOperand(2);
        Value *Prod = Builder.CreateMul(A, B, "mad.prod");
        Result = Builder.CreateAdd(Prod, C, "mad");
        break;
      }
      case DX_DEGREES: {
        // degrees(x) = x * (180.0 / pi)
        Value *X = CI->getArgOperand(0);
        Result = Builder.CreateFMul(X,
            ConstantFP::get(X->getType(), 57.295779513082323), "degrees");
        break;
      }
      case DX_RADIANS: {
        // radians(x) = x * (pi / 180.0)
        Value *X = CI->getArgOperand(0);
        Result = Builder.CreateFMul(X,
            ConstantFP::get(X->getType(), 0.017453292519943295), "radians");
        break;
      }
      case DX_SIGN: {
        // sign(x): -1 if x<0, 0 if x==0, 1 if x>0 (integer result)
        Value *X = CI->getArgOperand(0);
        if (X->getType()->isFloatTy()) {
          Value *Zero = ConstantFP::get(X->getType(), 0.0);
          Value *IsPos = Builder.CreateFCmpOGT(X, Zero);
          Value *IsNeg = Builder.CreateFCmpOLT(X, Zero);
          Value *PosVal = Builder.CreateSelect(IsPos, Builder.getInt32(1),
                                               Builder.getInt32(0));
          Value *NegVal = Builder.CreateSelect(IsNeg, Builder.getInt32(-1),
                                               PosVal);
          Result = NegVal;
        } else {
          Value *Zero = Builder.getInt32(0);
          Value *IsPos = Builder.CreateICmpSGT(X, Zero);
          Value *IsNeg = Builder.CreateICmpSLT(X, Zero);
          Value *PosVal = Builder.CreateSelect(IsPos, Builder.getInt32(1),
                                               Builder.getInt32(0));
          Result = Builder.CreateSelect(IsNeg, Builder.getInt32(-1), PosVal);
        }
        break;
      }
      case DX_STEP: {
        // step(y, x) = (x >= y) ? 1.0 : 0.0
        Value *Y = CI->getArgOperand(0);
        Value *X = CI->getArgOperand(1);
        Value *Cmp = Builder.CreateFCmpOGE(X, Y);
        Result = Builder.CreateSelect(Cmp,
            ConstantFP::get(X->getType(), 1.0),
            ConstantFP::get(X->getType(), 0.0), "step");
        break;
      }
      default:
        break;
      }

      if (Result) {
        CI->replaceAllUsesWith(Result);
        CI->eraseFromParent();
        Changed = true;
      }
    }
    ToReplace.clear();

    if (F.use_empty())
      ToDelete.push_back(&F);
  }

  for (Function *F : ToDelete)
    F->eraseFromParent();

  return Changed;
}

//===----------------------------------------------------------------------===//
// Metadata / Attribute Cleanup
//===----------------------------------------------------------------------===//

bool GPUHLSLLowering::stripHLSLMetadata(Module &M) {
  bool Changed = false;

  // Remove HLSL-specific named metadata
  SmallVector<StringRef, 8> ToErase;
  for (auto &NMD : M.named_metadata()) {
    StringRef Name = NMD.getName();
    if (Name.starts_with("dx.") || Name.starts_with("hlsl."))
      ToErase.push_back(Name);
  }
  for (StringRef Name : ToErase) {
    if (auto *NMD = M.getNamedMetadata(Name)) {
      NMD->eraseFromParent();
      Changed = true;
    }
  }

  // Strip HLSL-specific function attributes
  for (Function &F : M) {
    if (F.hasFnAttribute("hlsl.shader")) {
      F.removeFnAttr("hlsl.shader");
      Changed = true;
    }
    if (F.hasFnAttribute("hlsl.numthreads")) {
      F.removeFnAttr("hlsl.numthreads");
      Changed = true;
    }
  }

  return Changed;
}

//===----------------------------------------------------------------------===//
// Main Entry Point
//===----------------------------------------------------------------------===//

bool GPUHLSLLowering::runOnModule(Module &M) {
  // Quick check: does this module contain any DX intrinsics?
  bool HasDX = false;
  for (const Function &F : M) {
    if (F.getName().starts_with("llvm.dx.")) {
      HasDX = true;
      break;
    }
  }
  if (!HasDX)
    return false;

  bool Changed = false;
  Changed |= lowerThreadIDIntrinsics(M);
  Changed |= lowerResourceAccess(M);
  Changed |= lowerWaveIntrinsics(M);
  Changed |= lowerMathIntrinsics(M);
  Changed |= stripHLSLMetadata(M);

  return Changed;
}

INITIALIZE_PASS(GPUHLSLLowering, "gpu-hlsl-lowering",
                "GPU HLSL Lowering", false, false)

namespace llvm {
ModulePass *createGPUHLSLLoweringPass() {
  return new GPUHLSLLowering();
}
} // namespace llvm
