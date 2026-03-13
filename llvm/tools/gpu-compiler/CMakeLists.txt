set(LLVM_LINK_COMPONENTS
  Core
  GPUCodeGen
  GPUDesc
  GPUInfo
  IRReader
  MC
  Support
  Target
  TargetParser
)

add_llvm_tool(gpu-compiler
  gpu-compiler.cpp
)
