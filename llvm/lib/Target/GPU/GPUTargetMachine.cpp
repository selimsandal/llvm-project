//===-- GPUTargetMachine.cpp - GPU Target Machine ---------===//

#include "GPUTargetMachine.h"
#include "GPU.h"
#include "GPUSubtarget.h"
#include "TargetInfo/GPUTargetInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"

using namespace llvm;

// p1:32:32 = addrspace(1) pointers are 32-bit (OpenCL global memory)
static const char *GPUDataLayout = "e-p:32:32-p1:32:32-p3:32:32-i32:32-f32:32-n32";

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::Static);
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeGPUTarget() {
  RegisterTargetMachine<GPUTargetMachine> X(getTheGPUTarget());
  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeGPUDAGToDAGISelLegacyPass(PR);
  initializeGPUControlFlowPass(PR);
  initializeGPUPeepholePass(PR);
  initializeGPUSPIRVLoweringPass(PR);
  initializeGPUHLSLLoweringPass(PR);
}

GPUTargetMachine::GPUTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   std::optional<Reloc::Model> RM,
                                   std::optional<CodeModel::Model> CM,
                                   CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, GPUDataLayout, TT,
                               CPU.empty() ? "generic" : CPU, FS, Options,
                               getEffectiveRelocModel(RM),
                               CM.value_or(CodeModel::Small), OL),
      Subtarget(TT, CPU.empty() ? "generic" : std::string(CPU),
                std::string(FS), *this),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {
  initAsmInfo();
}

namespace {
class GPULocalMemoryGlobalsPass : public ModulePass {
public:
  static char ID;
  GPULocalMemoryGlobalsPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    static constexpr uint64_t GPULocalMemBytes = 4096ull * 4ull;
    const DataLayout &DL = M.getDataLayout();
    SmallVector<GlobalVariable *, 8> LocalGlobals;
    uint64_t NextOffset = 0;
    bool Changed = false;

    for (GlobalVariable &GV : M.globals()) {
      if (GV.getAddressSpace() != 3)
        continue;
      LocalGlobals.push_back(&GV);
    }

    for (GlobalVariable *GV : LocalGlobals) {
      if (GV->hasInitializer() && !GV->getInitializer()->isNullValue() &&
          !isa<UndefValue>(GV->getInitializer())) {
        report_fatal_error("GPU local addrspace globals only support zero/undef "
                           "initializers");
      }

      Align A = GV->getAlign().value_or(DL.getABITypeAlign(GV->getValueType()));
      NextOffset = alignTo(NextOffset, A);
      uint64_t Size = DL.getTypeAllocSize(GV->getValueType());
      if (NextOffset + Size > GPULocalMemBytes) {
        report_fatal_error("GPU local addrspace globals exceed per-workgroup "
                           "local memory capacity");
      }

      Constant *Offset =
          ConstantInt::get(Type::getInt32Ty(M.getContext()), NextOffset);
      Constant *Ptr = ConstantExpr::getIntToPtr(Offset, GV->getType());
      GV->replaceAllUsesWith(Ptr);
      NextOffset += Size;
      Changed = true;
    }

    for (GlobalVariable *GV : LocalGlobals) {
      if (GV->use_empty())
        GV->eraseFromParent();
    }

    return Changed;
  }
};

char GPULocalMemoryGlobalsPass::ID = 0;

static ModulePass *createGPULocalMemoryGlobalsPass() {
  return new GPULocalMemoryGlobalsPass();
}

class GPUPassConfig : public TargetPassConfig {
public:
  GPUPassConfig(GPUTargetMachine &TM, PassManagerBase *PM)
      : TargetPassConfig(TM, *PM) {}

  GPUTargetMachine &getGPUTargetMachine() const {
    return getTM<GPUTargetMachine>();
  }

  void addIRPasses() override {
    addPass(createGPUSPIRVLoweringPass());
    addPass(createGPUHLSLLoweringPass());
    addPass(createGPULocalMemoryGlobalsPass());
    // Keep addrspace(3) distinct so local/shared memory reaches target lowering
    // as real local-memory operations instead of being flattened to global.
    // Feed the backend already-structured CFG whenever possible so the late
    // machine control-flow pass only has to lower, not recover, structure.
    addPass(createFixIrreduciblePass());
    addPass(createUnifyLoopExitsPass());
    addPass(createStructurizeCFGPass(/*SkipUniformRegions=*/false));
    TargetPassConfig::addIRPasses();
  }

  bool addInstSelector() override {
    addPass(createGPUISelDag(getGPUTargetMachine(), getOptLevel()));
    return false;
  }

  void addPreEmitPass() override {
    addPass(createGPUControlFlowPass());
    addPass(createGPUPeepholePass());
  }
};
} // namespace

TargetPassConfig *GPUTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new GPUPassConfig(*this, &PM);
}
