add_llvm_component_library(LLVMGPUDesc
  GPUAsmBackend.cpp
  GPUInstPrinter.cpp
  GPUMCAsmInfo.cpp
  GPUMCCodeEmitter.cpp
  GPUMCTargetDesc.cpp

  LINK_COMPONENTS
  GPUInfo
  MC
  Support
  TargetParser

  ADD_TO_COMPONENT
  GPU
)
