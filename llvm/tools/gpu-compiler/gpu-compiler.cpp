//===-- gpu-compiler.cpp - Standalone GPU Compiler Tool ---===//
//
// Compiles LLVM IR (.ll) to raw GPU binary (.gpubin).
// Output is a flat binary of 128-bit instructions packed into
// 256-bit beats, ready for DMA upload via gpu_shader_loader.h.
//
// Usage:
//   gpu-compiler input.ll -o output.gpubin
//   gpu-compiler input.ll                     # writes to stdout
//
//===--------------------------------------------------------------===//

#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/IR/LegacyPassManager.h"

using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input .ll file>"), cl::Required);

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"),
               cl::init("-"));

static codegen::RegisterCodeGenFlags CGF;

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  // Initialize the GPU target
  LLVMInitializeGPUTargetInfo();
  LLVMInitializeGPUTarget();
  LLVMInitializeGPUTargetMC();
  LLVMInitializeGPUAsmPrinter();

  cl::ParseCommandLineOptions(argc, argv, "GPU Compiler\n");

  LLVMContext Context;
  SMDiagnostic Err;

  // Parse LLVM IR
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  // Set target triple
  Triple TT("gpu-none-none");
  M->setTargetTriple(TT);

  // Look up the GPU target
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
  if (!TheTarget) {
    errs() << argv[0] << ": " << Error << "\n";
    return 1;
  }

  // Create target machine
  TargetOptions Options = codegen::InitTargetOptionsFromCodeGenFlags(TT);
  std::string CPUStr = codegen::getCPUStr();
  std::unique_ptr<TargetMachine> TM(TheTarget->createTargetMachine(
      TT, CPUStr.empty() ? "generic" : CPUStr, "", Options, Reloc::Static));
  if (!TM) {
    errs() << argv[0] << ": could not create target machine\n";
    return 1;
  }

  M->setDataLayout(TM->createDataLayout());

  // Open output file
  std::error_code EC;
  auto Out = std::make_unique<ToolOutputFile>(OutputFilename, EC,
                                               sys::fs::OF_None);
  if (EC) {
    errs() << argv[0] << ": " << EC.message() << "\n";
    return 1;
  }

  // Run the code generation pipeline, emit object file
  legacy::PassManager PM;
  if (TM->addPassesToEmitFile(PM, Out->os(), nullptr,
                              CodeGenFileType::ObjectFile)) {
    errs() << argv[0] << ": target does not support object file emission\n";
    return 1;
  }

  PM.run(*M);
  Out->keep();

  return 0;
}
