include "llvm/Target/Target.td"

include "TempestRegisterInfo.td"
include "TempestInstrInfo.td"

def : ProcessorModel<"generic", NoSchedModel, []>;

def TempestInstrInfo : InstrInfo;

def Tempest : Target {
  let InstructionSet = TempestInstrInfo;
}