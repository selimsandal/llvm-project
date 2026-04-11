//===-- GPUFrameLowering.cpp - GPU Frame Lowering ---------===//

#include "GPUFrameLowering.h"
#include "GPUSubtarget.h"
#include "MCTargetDesc/GPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

using namespace llvm;

GPUFrameLowering::GPUFrameLowering(const GPUSubtarget &STI)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown,
                          Align(4), 0) {}

void GPUFrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  if (!MFI.getNumObjects())
    return;  // No spills — zero overhead

  // Compute frame size: number of spill slots * 4 bytes per slot
  uint64_t FrameSize = MFI.estimateStackSize(MF);
  if (FrameSize == 0)
    return;

  // Round up to multiple of 4
  FrameSize = (FrameSize + 3) & ~3ULL;
  MFI.setStackSize(FrameSize);

  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL;

  // The GPU has 4 compute engines that can each run a different
  // workgroup concurrently. Each engine has 8 SIMD lanes. The DDR
  // spill area at 0x00380000 must be partitioned per
  // (workgroup, lane) so concurrent workgroups on different engines
  // do not trample each other's spills.
  //
  // The compiler does not know an engine id at compile time, so we
  // use the linear group id instead — every concurrently-resident
  // workgroup has a distinct group id, so the resulting stack region
  // is unique:
  //
  //   linear_gid = group_id_x
  //              + group_id_y * num_groups_x
  //              + group_id_z * num_groups_x * num_groups_y
  //
  //   r30 = 0x00380000 + (linear_gid * 8 + lane) * frame_size
  //
  // For kernels that genuinely run as a single workgroup (group_id_*
  // and num_groups_* all default to 0/1) the formula degenerates to
  // `lane * frame_size`, matching the single-workgroup behaviour the
  // backend used to emit unconditionally.
  //
  // Scratch registers used by this prologue: r25..r29 and r30. At
  // function entry only r0 (lane id) and r1..r4 (kernel args) are
  // live-in, so r5..r29 are free to clobber.
  constexpr unsigned SREG_GROUP_ID_X   = 48;
  constexpr unsigned SREG_GROUP_ID_Y   = 49;
  constexpr unsigned SREG_GROUP_ID_Z   = 50;
  constexpr unsigned SREG_NUM_GROUPS_X = 54;
  constexpr unsigned SREG_NUM_GROUPS_Y = 55;
  constexpr uint64_t SPILL_BASE        = 0x00380000ULL;

  // r25..r29 = group_id_x/y/z, num_groups_x/y
  BuildMI(MBB, MBBI, DL, TII.get(GPU::GETSR), GPU::R25)
      .addImm(SREG_GROUP_ID_X);
  BuildMI(MBB, MBBI, DL, TII.get(GPU::GETSR), GPU::R26)
      .addImm(SREG_GROUP_ID_Y);
  BuildMI(MBB, MBBI, DL, TII.get(GPU::GETSR), GPU::R27)
      .addImm(SREG_GROUP_ID_Z);
  BuildMI(MBB, MBBI, DL, TII.get(GPU::GETSR), GPU::R28)
      .addImm(SREG_NUM_GROUPS_X);
  BuildMI(MBB, MBBI, DL, TII.get(GPU::GETSR), GPU::R29)
      .addImm(SREG_NUM_GROUPS_Y);

  // r30 = group_id_y * num_groups_x
  BuildMI(MBB, MBBI, DL, TII.get(GPU::MUL), GPU::R30)
      .addReg(GPU::R26)
      .addReg(GPU::R28);
  // r25 = group_id_x + (gy * ngx)
  BuildMI(MBB, MBBI, DL, TII.get(GPU::ADD), GPU::R25)
      .addReg(GPU::R25)
      .addReg(GPU::R30);
  // r30 = num_groups_x * num_groups_y
  BuildMI(MBB, MBBI, DL, TII.get(GPU::MUL), GPU::R30)
      .addReg(GPU::R28)
      .addReg(GPU::R29);
  // r27 = group_id_z * (ngx * ngy)
  BuildMI(MBB, MBBI, DL, TII.get(GPU::MUL), GPU::R27)
      .addReg(GPU::R27)
      .addReg(GPU::R30);
  // r25 = linear_gid = (gx + gy*ngx) + gz*ngx*ngy
  BuildMI(MBB, MBBI, DL, TII.get(GPU::ADD), GPU::R25)
      .addReg(GPU::R25)
      .addReg(GPU::R27);
  // r25 = linear_gid * 8  (one slab per workgroup, 8 lanes wide)
  BuildMI(MBB, MBBI, DL, TII.get(GPU::SHLi), GPU::R25)
      .addReg(GPU::R25)
      .addImm(3);
  // r25 = (linear_gid * 8) + lane
  BuildMI(MBB, MBBI, DL, TII.get(GPU::ADD), GPU::R25)
      .addReg(GPU::R25)
      .addReg(GPU::R0);
  // r30 = ((linear_gid * 8) + lane) * frame_size
  BuildMI(MBB, MBBI, DL, TII.get(GPU::MULi), GPU::R30)
      .addReg(GPU::R25)
      .addImm(FrameSize);
  // r30 += spill base
  BuildMI(MBB, MBBI, DL, TII.get(GPU::ADDi), GPU::R30)
      .addReg(GPU::R30)
      .addImm(SPILL_BASE);
}

void GPUFrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {}

bool GPUFrameLowering::hasFPImpl(const MachineFunction &MF) const {
  return false;
}

MachineBasicBlock::iterator GPUFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MI) const {
  return MBB.erase(MI);
}
