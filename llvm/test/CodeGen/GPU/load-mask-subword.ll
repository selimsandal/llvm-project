; RUN: llc -march=gpu < %s | FileCheck %s

; Regression for: `llc -march=gpu` used to crash in
; `GPUTargetLowering::LowerBR_CC` whenever DAGCombine merged
; `load i32 + and 0xff` (or `0xffff`) into an i8/i16 extload.
; The fix is twofold:
;   - extload actions are Expand (not Promote) in GPUISelLowering
;   - shouldReduceLoadWidth refuses i1/i8/i16 narrowing so DAGCombiner
;     never introduces the extload in the first place
;
; The important thing here is that llc does not crash and emits both
; a plain 32-bit load and the AND mask, without any sub-word load.

define void @mask_ff(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: mask_ff:
; CHECK:       ld_scatter
; CHECK:       and{{[[:space:]]}}{{.*}}0xff
; CHECK:       st_scatter
; CHECK:       halt
  %v = load i32, ptr addrspace(1) %in, align 4
  %m = and i32 %v, 255
  store i32 %m, ptr addrspace(1) %out, align 4
  ret void
}

define void @mask_ffff(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: mask_ffff:
; CHECK:       ld_scatter
; CHECK:       and{{[[:space:]]}}{{.*}}0xffff
; CHECK:       st_scatter
; CHECK:       halt
  %v = load i32, ptr addrspace(1) %in, align 4
  %m = and i32 %v, 65535
  store i32 %m, ptr addrspace(1) %out, align 4
  ret void
}

; The same pattern feeding a local-memory store index — this was the
; original shape that crashed (`llc` segfaulted in LowerBR_CC while
; synthesizing a BR_CC node during legalization of the addrspace(3) store).

@lm = internal addrspace(3) global [256 x i32] undef, align 4

define void @mem_indexed_local_store(ptr addrspace(1) %in) {
; CHECK-LABEL: mem_indexed_local_store:
; CHECK:       ld_scatter
; CHECK:       and{{[[:space:]]}}{{.*}}0xff
; CHECK:       st_local
; CHECK:       halt
  %v = load i32, ptr addrspace(1) %in, align 4
  %idx = and i32 %v, 255
  %gep = getelementptr inbounds [256 x i32], ptr addrspace(3) @lm, i32 0, i32 %idx
  store i32 42, ptr addrspace(3) %gep, align 4
  ret void
}
