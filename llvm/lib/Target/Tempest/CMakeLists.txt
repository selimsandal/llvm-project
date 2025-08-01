add_llvm_component_group(Tempest)

set(LLVM_TARGET_DEFINITIONS Tempest.td)

tablegen(LLVM TempestGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM TempestGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM TempestGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM TempestGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM TempestGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(TempestCommonTableGen)

add_llvm_target(TempestCodeGen
  TempestTargetMachine.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  Core
  MC
  SelectionDAG
  Support
  Target
  TargetParser
  TempestDesc
  TempestInfo

  ADD_TO_COMPONENT
  Tempest
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)