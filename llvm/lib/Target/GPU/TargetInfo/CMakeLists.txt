add_llvm_component_library(LLVMGPUInfo
  GPUTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  GPU
)
