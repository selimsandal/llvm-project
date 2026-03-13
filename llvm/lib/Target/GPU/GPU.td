//===-- GPU.td - Top-level GPU target definition ----*- tablegen -*-===//
//
// Main TableGen file for the custom GPU backend.
//
//===--------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===--------------------------------------------------------------===//
// GPU subtarget features and processors
//===--------------------------------------------------------------===//

def FeatureScoreboard : SubtargetFeature<"scoreboard", "HasScoreboard",
  "true", "Enable hardware scoreboard for non-blocking ops">;

def FeatureBarrelThreading : SubtargetFeature<"barrel", "HasBarrelThreading",
  "true", "Enable barrel threading">;

//===--------------------------------------------------------------===//
// Register file, calling conv, scheduling, instruction descriptions
//===--------------------------------------------------------------===//

include "GPUSchedule.td"
include "GPURegisterInfo.td"
include "GPUCallingConv.td"
include "GPUInstrInfo.td"

// Remap pseudo-pointer operands to GPR class
defm : RemapAllTargetPseudoPointerOperands<GPR>;

def GPUInstrInfo : InstrInfo;

//===--------------------------------------------------------------===//
// Processors
//===--------------------------------------------------------------===//

def : ProcessorModel<"generic", GPUSchedModel, []>;
def : ProcessorModel<"xe3", GPUSchedModel, [FeatureScoreboard]>;

//===--------------------------------------------------------------===//
// Assembly printer
//===--------------------------------------------------------------===//

def GPUInstPrinter : AsmWriter {
  string AsmWriterClassName = "InstPrinter";
  bit isMCAsmWriter = 1;
}

//===--------------------------------------------------------------===//
// Target declaration
//===--------------------------------------------------------------===//

def GPU : Target {
  let InstructionSet = GPUInstrInfo;
  let AssemblyWriters = [GPUInstPrinter];
}
