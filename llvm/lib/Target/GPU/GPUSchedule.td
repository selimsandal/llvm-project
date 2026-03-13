//===-- GPUSchedule.td - GPU Scheduling Model ---*- tablegen -*-===//
//
// Scheduling model matching hardware latencies from gpu_compute_engine.sv.
//
//===--------------------------------------------------------------===//

def GPUSchedModel : SchedMachineModel {
  let IssueWidth = 1;        // Single-issue, in-order
  let MicroOpBufferSize = 0; // In-order
  let LoadLatency = 10;      // LDV latency
  let MispredictPenalty = 0;  // No branch prediction
  let CompleteModel = 0;     // Allow unlisted instructions
}

// Define scheduling resources
def ALU    : ProcResource<1>;
def MULP   : ProcResource<1>;
def FPU    : ProcResource<1>;
def MEMPORT: ProcResource<1>;

// Scheduling classes
def WriteALU : SchedWrite;
def WriteMUL : SchedWrite;
def WriteFPU : SchedWrite;
def WriteFMA : SchedWrite;
def WriteMem : SchedWrite;

let SchedModel = GPUSchedModel in {
  def : WriteRes<WriteALU, [ALU]> { let Latency = 1; }
  def : WriteRes<WriteMUL, [MULP]> { let Latency = 2; }
  def : WriteRes<WriteFPU, [FPU]> { let Latency = 4; }
  def : WriteRes<WriteFMA, [FPU]> { let Latency = 5; }
  def : WriteRes<WriteMem, [MEMPORT]> { let Latency = 10; }
}
