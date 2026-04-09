//===-- GPUKernelMetadata.cpp - Emit GPU Kernel Metadata ------------------===//
//
// Emits a small .gpu.meta ELF section alongside .text so the host runtime can
// discover the kernel's launch requirements without duplicating them in host
// code.
//
// The v1 record intentionally stays small:
//   - required local size if the frontend declared one
//   - whether the kernel needs the hidden workgroup-context beat
//   - basic feature bits for local memory / local atomics / barriers
//   - enough argument info to validate direct local-pointer args
//
//===----------------------------------------------------------------------===//

#include "GPU.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-kernel-metadata"

namespace {

enum GPUKernelMetaFlags : uint32_t {
  GPU_META_FLAG_HAS_REQD_LOCAL_SIZE = 1u << 0,
  GPU_META_FLAG_NEEDS_WG_CTX        = 1u << 1,
  GPU_META_FLAG_USES_LOCAL_MEMORY   = 1u << 2,
  GPU_META_FLAG_USES_LOCAL_ATOMICS  = 1u << 3,
  GPU_META_FLAG_USES_BARRIER        = 1u << 4,
  GPU_META_FLAG_USES_MEM_FENCE      = 1u << 5,
  GPU_META_FLAG_FRONTEND_OPENCL     = 1u << 6,
  GPU_META_FLAG_FRONTEND_HLSL       = 1u << 7,
  GPU_META_FLAG_INDIRECT_ARGS       = 1u << 8,
};

struct GPUKernelMetaInfo {
  uint32_t Flags = 0;
  uint16_t LocalSizeX = 0;
  uint16_t LocalSizeY = 0;
  uint16_t LocalSizeZ = 0;
  uint16_t ArgCount = 0;
  uint32_t DirectLocalArgMask = 0;
};

class GPUKernelMetadata : public ModulePass {
public:
  static char ID;
  GPUKernelMetadata() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override {
    return "GPU Kernel Metadata";
  }

private:
  Function *findKernelEntry(Module &M) const;
  bool collectMetadata(Function &Entry, GPUKernelMetaInfo &Info) const;
  bool emitMetadata(Module &M, const GPUKernelMetaInfo &Info) const;
};

char GPUKernelMetadata::ID = 0;

static constexpr uint32_t GPUKernelMetaMagic = 0x4D555047u; // "GPUM"
static constexpr uint16_t GPUKernelMetaVersion = 1u;
static constexpr uint16_t GPUKernelMetaSizeBytes = 24u;

static uint16_t getKernelArgSlotWidth(const Argument &Arg) {
  Type *Ty = Arg.getType();
  if (Ty->isIntegerTy(64))
    return 2;
  return 1;
}

static bool parseUInt16(StringRef Text, uint16_t &Out) {
  unsigned Value = 0;
  if (Text.getAsInteger(10, Value) || Value > 0xFFFFu)
    return false;
  Out = static_cast<uint16_t>(Value);
  return true;
}

static bool parseHLSLNumThreads(Function &F,
                                uint16_t &X, uint16_t &Y, uint16_t &Z) {
  Attribute Attr = F.getFnAttribute("hlsl.numthreads");
  if (!Attr.isValid())
    return false;

  SmallVector<StringRef, 3> Parts;
  Attr.getValueAsString().split(Parts, ',', -1, false);
  if (Parts.size() != 3)
    return false;

  return parseUInt16(Parts[0].trim(), X) &&
         parseUInt16(Parts[1].trim(), Y) &&
         parseUInt16(Parts[2].trim(), Z);
}

static bool parseReqdWorkGroupSize(Function &F,
                                   uint16_t &X, uint16_t &Y, uint16_t &Z) {
  MDNode *MD = F.getMetadata("reqd_work_group_size");
  if (!MD || MD->getNumOperands() != 3)
    return false;

  uint16_t Values[3] = {};
  for (unsigned I = 0; I < 3; ++I) {
    auto *CMD = dyn_cast<ConstantAsMetadata>(MD->getOperand(I));
    auto *CI = CMD ? dyn_cast<ConstantInt>(CMD->getValue()) : nullptr;
    if (!CI || CI->getZExtValue() > 0xFFFFu)
      return false;
    Values[I] = static_cast<uint16_t>(CI->getZExtValue());
  }

  X = Values[0];
  Y = Values[1];
  Z = Values[2];
  return true;
}

static unsigned getPointerAddressSpace(Value *V) {
  auto *PT = dyn_cast<PointerType>(V->getType());
  return PT ? PT->getAddressSpace() : ~0u;
}

static unsigned getKernelArgAddressSpace(Function &F, unsigned Index) {
  if (MDNode *MD = F.getMetadata("kernel_arg_addr_space")) {
    if (Index < MD->getNumOperands()) {
      auto *CMD = dyn_cast<ConstantAsMetadata>(MD->getOperand(Index));
      auto *CI = CMD ? dyn_cast<ConstantInt>(CMD->getValue()) : nullptr;
      if (CI)
        return static_cast<unsigned>(CI->getZExtValue());
    }
  }

  auto ArgIt = F.arg_begin();
  std::advance(ArgIt, Index);
  return getPointerAddressSpace(&*ArgIt);
}

Function *GPUKernelMetadata::findKernelEntry(Module &M) const {
  Function *OpenCLKernel = nullptr;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (F.hasFnAttribute("hlsl.shader"))
      return &F;
    if (!OpenCLKernel &&
        (F.getMetadata("kernel_arg_addr_space") ||
         F.getMetadata("reqd_work_group_size"))) {
      OpenCLKernel = &F;
    }
  }

  return OpenCLKernel;
}

bool GPUKernelMetadata::collectMetadata(Function &Entry,
                                        GPUKernelMetaInfo &Info) const {
  bool HasMetadata = false;

  if (Entry.hasFnAttribute("hlsl.shader")) {
    Info.Flags |= GPU_META_FLAG_FRONTEND_HLSL;
    HasMetadata = true;
  } else if (Entry.getMetadata("kernel_arg_addr_space") ||
             Entry.getMetadata("reqd_work_group_size")) {
    Info.Flags |= GPU_META_FLAG_FRONTEND_OPENCL;
    HasMetadata = true;
  }

  uint16_t X = 0, Y = 0, Z = 0;
  if (parseHLSLNumThreads(Entry, X, Y, Z) ||
      parseReqdWorkGroupSize(Entry, X, Y, Z)) {
    Info.Flags |= GPU_META_FLAG_HAS_REQD_LOCAL_SIZE;
    Info.LocalSizeX = X;
    Info.LocalSizeY = Y;
    Info.LocalSizeZ = Z;
    HasMetadata = true;
  }

  {
    for (Argument &Arg : Entry.args())
      Info.ArgCount = static_cast<uint16_t>(
          Info.ArgCount + getKernelArgSlotWidth(Arg));
  }
  if (Info.ArgCount > 4)
    Info.Flags |= GPU_META_FLAG_INDIRECT_ARGS;

  // Emit the local-arg mask for both the direct (<=4 args) and indirect
  // (>4 args) paths. The host needs to know which slots in the indirect
  // arg buffer should be packed as `__local` byte offsets vs DDR pointers,
  // and the compiler is the only place that knows the original kernel
  // address-space metadata. The mask is limited to 32 args.
  {
    unsigned SourceIndex = 0;
    unsigned SlotIndex = 0;
    for (Argument &Arg : Entry.args()) {
      if (SlotIndex < 32 && getKernelArgAddressSpace(Entry, SourceIndex) == 3)
        Info.DirectLocalArgMask |= (1u << SlotIndex);
      SlotIndex += getKernelArgSlotWidth(Arg);
      ++SourceIndex;
    }
  }

  if (Info.DirectLocalArgMask)
    Info.Flags |= GPU_META_FLAG_USES_LOCAL_MEMORY;

  for (Instruction &I : instructions(Entry)) {
    if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (Function *Callee = CB->getCalledFunction()) {
        switch (Callee->getIntrinsicID()) {
        case Intrinsic::gpu_getsr:
          Info.Flags |= GPU_META_FLAG_NEEDS_WG_CTX;
          HasMetadata = true;
          break;
        case Intrinsic::gpu_workgroup_sync:
          Info.Flags |= GPU_META_FLAG_USES_BARRIER;
          HasMetadata = true;
          break;
        case Intrinsic::gpu_mem_fence:
          Info.Flags |= GPU_META_FLAG_USES_MEM_FENCE;
          HasMetadata = true;
          break;
        default:
          break;
        }
      }
    }

    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (getPointerAddressSpace(LI->getPointerOperand()) == 3)
        Info.Flags |= GPU_META_FLAG_USES_LOCAL_MEMORY;
    } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (getPointerAddressSpace(SI->getPointerOperand()) == 3)
        Info.Flags |= GPU_META_FLAG_USES_LOCAL_MEMORY;
    } else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
      if (getPointerAddressSpace(RMW->getPointerOperand()) == 3) {
        Info.Flags |= GPU_META_FLAG_USES_LOCAL_MEMORY;
        Info.Flags |= GPU_META_FLAG_USES_LOCAL_ATOMICS;
      }
    } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) {
      if (getPointerAddressSpace(CX->getPointerOperand()) == 3) {
        Info.Flags |= GPU_META_FLAG_USES_LOCAL_MEMORY;
        Info.Flags |= GPU_META_FLAG_USES_LOCAL_ATOMICS;
      }
    }
  }

  return HasMetadata;
}

bool GPUKernelMetadata::emitMetadata(Module &M,
                                     const GPUKernelMetaInfo &Info) const {
  if (M.getGlobalVariable("__gpu_kernel_metadata"))
    return false;

  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I16 = Type::getInt16Ty(Ctx);
  StructType *MetaTy = StructType::get(
      Ctx, {I32, I16, I16, I32, I16, I16, I16, I16, I32});

  Constant *Init = ConstantStruct::get(
      MetaTy,
      ConstantInt::get(I32, GPUKernelMetaMagic),
      ConstantInt::get(I16, GPUKernelMetaVersion),
      ConstantInt::get(I16, GPUKernelMetaSizeBytes),
      ConstantInt::get(I32, Info.Flags),
      ConstantInt::get(I16, Info.LocalSizeX),
      ConstantInt::get(I16, Info.LocalSizeY),
      ConstantInt::get(I16, Info.LocalSizeZ),
      ConstantInt::get(I16, Info.ArgCount),
      ConstantInt::get(I32, Info.DirectLocalArgMask));

  auto *GV = new GlobalVariable(M, MetaTy, /*isConstant=*/true,
                                GlobalValue::PrivateLinkage, Init,
                                "__gpu_kernel_metadata");
  GV->setSection(".gpu.meta");
  GV->setAlignment(Align(4));
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  appendToCompilerUsed(M, ArrayRef<GlobalValue *>{GV});
  return true;
}

bool GPUKernelMetadata::runOnModule(Module &M) {
  Function *Entry = findKernelEntry(M);
  if (!Entry)
    return false;

  GPUKernelMetaInfo Info;
  if (!collectMetadata(*Entry, Info))
    return false;

  return emitMetadata(M, Info);
}

} // namespace

INITIALIZE_PASS(GPUKernelMetadata, "gpu-kernel-metadata",
                "GPU Kernel Metadata", false, false)

ModulePass *llvm::createGPUKernelMetadataPass() {
  return new GPUKernelMetadata();
}
