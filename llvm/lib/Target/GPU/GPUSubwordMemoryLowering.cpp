//===-- GPUSubwordMemoryLowering.cpp - Sub-word global mem rewriter ===//
//
// IR-level pass that rewrites sub-word (i1/i8/i16) loads and stores in
// the global address space (addrspace(1)) into aligned i32 word
// load/word read-modify-write sequences. The GPU has only 32-bit
// LD_SCATTER / ST_SCATTER memory ops; without this rewrite a kernel
// using `__global char*` (e.g. Rodinia bfs) reaches ISel as an
// `EXTLOAD anyext from i8` that the backend has no pattern for and
// crashes "Cannot select" out of SelectionDAGISel.
//
// Loads are exact: align the pointer down to a word boundary, do a
// regular i32 load, shift right by `(addr & 3) * 8`, then mask /
// sign-extend per the requested extension.
//
// Stores are lowered through a 32-bit cmpxchg loop: load the aligned
// word, clear the byte slot with an AND mask, OR in the new value
// shifted into place, then atomically swap the new word into memory.
// If another lane updated the word first, retry. This keeps the
// sub-word store semantics correct across lanes while still targeting
// the existing 32-bit global memory path.
//
// addrspace(3) (local memory) is intentionally left alone — the local
// memory backend has its own LD_LOCAL/ST_LOCAL path and the
// `GPULocalMemoryGlobalsPass` already runs against it.
//
//===--------------------------------------------------------------===//

#include "GPU.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-subword-memory-lowering"

namespace {

class GPUSubwordMemoryLowering : public ModulePass {
public:
  static char ID;
  GPUSubwordMemoryLowering() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override {
    return "GPU Subword Global Memory Lowering";
  }
};

char GPUSubwordMemoryLowering::ID = 0;

// Returns true if Ty is an integer type narrower than 32 bits.
static bool isSubwordIntType(Type *Ty) {
  if (auto *ITy = dyn_cast<IntegerType>(Ty))
    return ITy->getBitWidth() < 32 && ITy->getBitWidth() >= 1;
  return false;
}

// Returns true if the memory access is in a global-style address space
// that the backend lowers to LD_SCATTER / ST_SCATTER.
//   addrspace(0)  generic flat (treated as global on this target)
//   addrspace(1)  __global
// addrspace(3) (__local) is excluded — local memory has its own path.
static bool isGlobalAddrSpace(unsigned AS) { return AS == 0 || AS == 1; }

static bool isUnsupportedStore(StoreInst *SI) {
  return SI->isAtomic() || SI->isVolatile();
}

// Build the aligned-word base pointer + the byte offset within the word.
//   base    = inttoptr ((ptrtoint p) & ~3) to ptr addrspace(AS)
//   byteOff = (ptrtoint p) & 3
static void buildAlignedAddress(IRBuilder<> &B, Value *Ptr,
                                Value *&AlignedBase, Value *&ByteOff) {
  Type *I32 = B.getInt32Ty();
  unsigned AS = Ptr->getType()->getPointerAddressSpace();

  Value *PtrInt = B.CreatePtrToInt(Ptr, I32, "subword.addr");
  Value *Aligned = B.CreateAnd(PtrInt, ConstantInt::get(I32, ~3u),
                               "subword.addr.aligned");
  Value *Off = B.CreateAnd(PtrInt, ConstantInt::get(I32, 3u),
                           "subword.byte.off");

  AlignedBase = B.CreateIntToPtr(
      Aligned, PointerType::get(B.getContext(), AS), "subword.word.ptr");
  ByteOff = Off;
}

// Replace a sub-word load with an aligned i32 load + shift + mask.
static void rewriteLoad(LoadInst *LI) {
  IRBuilder<> B(LI);
  Value *Ptr = LI->getPointerOperand();

  Value *WordPtr;
  Value *ByteOff;
  buildAlignedAddress(B, Ptr, WordPtr, ByteOff);

  Type *I32 = B.getInt32Ty();
  LoadInst *Word = B.CreateAlignedLoad(I32, WordPtr, Align(4),
                                       LI->isVolatile(), "subword.word");
  Word->setAtomic(LI->getOrdering(), LI->getSyncScopeID());

  Value *BitShift = B.CreateShl(ByteOff, ConstantInt::get(I32, 3),
                                "subword.bit.off");
  Value *Shifted = B.CreateLShr(Word, BitShift, "subword.shifted");

  Type *Ty = LI->getType();
  unsigned BW = cast<IntegerType>(Ty)->getBitWidth();
  Value *Mask = ConstantInt::get(I32, (uint64_t(1) << BW) - 1);
  Value *Masked = B.CreateAnd(Shifted, Mask, "subword.masked");
  Value *Trunc = B.CreateTrunc(Masked, Ty, "subword.value");

  LI->replaceAllUsesWith(Trunc);
  LI->eraseFromParent();
}

static AtomicOrdering getCASSuccessOrdering(const StoreInst *SI) {
  return SI->isAtomic() ? SI->getOrdering() : AtomicOrdering::Monotonic;
}

static AtomicOrdering getCASFailureOrdering(AtomicOrdering SuccessOrdering) {
  return AtomicCmpXchgInst::getStrongestFailureOrdering(SuccessOrdering);
}

// Replace a sub-word store with an aligned i32 cmpxchg loop so stores
// to different bytes in the same word remain correct across lanes.
static void rewriteStore(StoreInst *SI) {
  BasicBlock *OrigBB = SI->getParent();
  Function *F = OrigBB->getParent();
  LLVMContext &Ctx = F->getContext();
  Value *Ptr = SI->getPointerOperand();
  Value *Val = SI->getValueOperand();

  BasicBlock *ContBB =
      OrigBB->splitBasicBlock(BasicBlock::iterator(SI), "subword.store.cont");
  OrigBB->getTerminator()->eraseFromParent();

  BasicBlock *LoopBB =
      BasicBlock::Create(Ctx, "subword.store.cas", F, ContBB);
  UncondBrInst::Create(LoopBB, OrigBB);

  IRBuilder<> B(LoopBB);

  Value *WordPtr;
  Value *ByteOff;
  buildAlignedAddress(B, Ptr, WordPtr, ByteOff);

  Type *I32 = B.getInt32Ty();
  unsigned BW = cast<IntegerType>(Val->getType())->getBitWidth();
  Value *FieldMask = ConstantInt::get(I32, (uint64_t(1) << BW) - 1);

  Value *BitShift = B.CreateShl(ByteOff, ConstantInt::get(I32, 3),
                                "subword.bit.off");
  Value *Shifted = B.CreateShl(FieldMask, BitShift, "subword.field.mask");
  Value *ClearMask = B.CreateNot(Shifted, "subword.clear.mask");

  AtomicOrdering SuccessOrdering = getCASSuccessOrdering(SI);
  AtomicOrdering FailureOrdering = getCASFailureOrdering(SuccessOrdering);

  LoadInst *OldWord = B.CreateAlignedLoad(I32, WordPtr, Align(4),
                                          SI->isVolatile(), "subword.old");
  Value *Cleared = B.CreateAnd(OldWord, ClearMask, "subword.cleared");

  Value *NewVal = B.CreateZExt(Val, I32, "subword.zext");
  NewVal = B.CreateAnd(NewVal, FieldMask, "subword.zext.masked");
  NewVal = B.CreateShl(NewVal, BitShift, "subword.zext.shifted");

  Value *NewWord = B.CreateOr(Cleared, NewVal, "subword.new");
  AtomicCmpXchgInst *CAS = B.CreateAtomicCmpXchg(
      WordPtr, OldWord, NewWord, Align(4), SuccessOrdering, FailureOrdering,
      SI->getSyncScopeID());
  CAS->setVolatile(SI->isVolatile());

  Value *Success = B.CreateExtractValue(CAS, 1, "subword.cas.success");
  B.CreateCondBr(Success, ContBB, LoopBB);

  SI->eraseFromParent();
}

bool GPUSubwordMemoryLowering::runOnModule(Module &M) {
  bool Changed = false;
  SmallVector<LoadInst *, 16> Loads;
  SmallVector<StoreInst *, 16> Stores;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (Instruction &I : instructions(F)) {
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (isSubwordIntType(LI->getType()) &&
            isGlobalAddrSpace(LI->getPointerAddressSpace()))
          Loads.push_back(LI);
        continue;
      }
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        Type *VTy = SI->getValueOperand()->getType();
        if (isSubwordIntType(VTy) &&
            isGlobalAddrSpace(SI->getPointerAddressSpace())) {
          if (isUnsupportedStore(SI)) {
            M.getContext().emitError(
                SI,
                "GPU backend does not support atomic or volatile sub-word "
                "global stores; use 32-bit accesses or full-word atomics");
            return false;
          }
          Stores.push_back(SI);
        }
        continue;
      }
    }
  }

  for (LoadInst *LI : Loads) {
    rewriteLoad(LI);
    Changed = true;
  }
  for (StoreInst *SI : Stores) {
    rewriteStore(SI);
    Changed = true;
  }

  return Changed;
}

} // anonymous namespace

INITIALIZE_PASS(GPUSubwordMemoryLowering, "gpu-subword-memory-lowering",
                "GPU Subword Global Memory Lowering", false, false)

namespace llvm {
ModulePass *createGPUSubwordMemoryLoweringPass() {
  return new GPUSubwordMemoryLowering();
}
} // namespace llvm
