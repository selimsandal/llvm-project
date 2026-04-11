add_llvm_component_group(GPU)

set(LLVM_TARGET_DEFINITIONS GPU.td)

tablegen(LLVM GPUGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM GPUGenCallingConv.inc -gen-callingconv)
tablegen(LLVM GPUGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM GPUGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM GPUGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM GPUGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM GPUGenSDNodeInfo.inc -gen-sd-node-info)
tablegen(LLVM GPUGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(GPUCommonTableGen)

add_llvm_target(GPUCodeGen
  GPUArgRegCopy.cpp
  GPUAsmPrinter.cpp
  GPUControlFlow.cpp
  GPUFrameLowering.cpp
  GPUHLSLLowering.cpp
  GPUInstrInfo.cpp
  GPUISelDAGToDAG.cpp
  GPUISelLowering.cpp
  GPUKernelMetadata.cpp
  GPUMCInstLower.cpp
  GPUPeephole.cpp
  GPURegisterInfo.cpp
  GPUSPIRVLowering.cpp
  GPUSubtarget.cpp
  GPUSubwordMemoryLowering.cpp
  GPUTargetMachine.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  GPUDesc
  GPUInfo
  MC
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  GPU
)

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
