// Tempest GPU Registers - Per Vector Processor
// Each of the 16 vector processors has this register set
class TempestSpecialReg<string n> : Register<n> {
    let Namespace = "Tempest";
}

class TempestGPRReg<string n> : Register<n> {
    let Namespace = "Tempest";
}

class TempestVecReg<string n> : Register<n> {
    let Namespace = "Tempest";
}

let Namespace = "Tempest" in {
  //== Per-Processor General Purpose Registers ==//
  //=============================================//
  def GPR0  : TempestGPRReg<"gpr0"> {
    let isConstant = 1;  // Zero register
  }
  def GPR1  : TempestGPRReg<"gpr1">;
  def GPR2  : TempestGPRReg<"gpr2">;
  def GPR3  : TempestGPRReg<"gpr3">;
  def GPR4  : TempestGPRReg<"gpr4">;
  def GPR5  : TempestGPRReg<"gpr5">;
  def GPR6  : TempestGPRReg<"gpr6">;
  def GPR7  : TempestGPRReg<"gpr7">;
  def GPR8  : TempestGPRReg<"gpr8">;
  def GPR9  : TempestGPRReg<"gpr9">;
  def GPR10 : TempestGPRReg<"gpr10">;
  def GPR11 : TempestGPRReg<"gpr11">;
  def GPR12 : TempestGPRReg<"gpr12">;
  def GPR13 : TempestGPRReg<"gpr13">;
  def GPR14 : TempestGPRReg<"gpr14">;
  def GPR15 : TempestGPRReg<"gpr15">;
  
  //== Per-Processor Special Registers ==//
  //=====================================//
  def PC       : TempestSpecialReg<"pc">;        // Program counter
  def TID      : TempestSpecialReg<"tid">;       // Thread/Processor ID (0-15)
  
  //== Per-Processor Vector Registers ==//
  //====================================//
  def VR0  : TempestVecReg<"vr0">;
  def VR1  : TempestVecReg<"vr1">;
  def VR2  : TempestVecReg<"vr2">;
  def VR3  : TempestVecReg<"vr3">;
  def VR4  : TempestVecReg<"vr4">;
  def VR5  : TempestVecReg<"vr5">;
  def VR6  : TempestVecReg<"vr6">;
  def VR7  : TempestVecReg<"vr7">;
  def VR8  : TempestVecReg<"vr8">;
  def VR9  : TempestVecReg<"vr9">;
  def VR10 : TempestVecReg<"vr10">;
  def VR11 : TempestVecReg<"vr11">;
  def VR12 : TempestVecReg<"vr12">;
  def VR13 : TempestVecReg<"vr13">;
  def VR14 : TempestVecReg<"vr14">;
  def VR15 : TempestVecReg<"vr15">;
  def VR16 : TempestVecReg<"vr16">;
  def VR17 : TempestVecReg<"vr17">;
  def VR18 : TempestVecReg<"vr18">;
  def VR19 : TempestVecReg<"vr19">;
  def VR20 : TempestVecReg<"vr20">;
  def VR21 : TempestVecReg<"vr21">;
  def VR22 : TempestVecReg<"vr22">;
  def VR23 : TempestVecReg<"vr23">;
  def VR24 : TempestVecReg<"vr24">;
  def VR25 : TempestVecReg<"vr25">;
  def VR26 : TempestVecReg<"vr26">;
  def VR27 : TempestVecReg<"vr27">;
  def VR28 : TempestVecReg<"vr28">;
  def VR29 : TempestVecReg<"vr29">;
  def VR30 : TempestVecReg<"vr30">;
  def VR31 : TempestVecReg<"vr31">;
}

// Register classes - these represent the registers available
// to a single vector processor
def GPR : RegisterClass<"Tempest", [i64], 64, (add
  GPR0,  GPR1,  GPR2,  GPR3,  GPR4,  GPR5,  GPR6,  GPR7,
  GPR8,  GPR9,  GPR10, GPR11, GPR12, GPR13, GPR14, GPR15
)>;

def VecReg : RegisterClass<"Tempest", [v16f32, v8f64], 512, (add
  VR0,  VR1,  VR2,  VR3,  VR4,  VR5,  VR6,  VR7,
  VR8,  VR9,  VR10, VR11, VR12, VR13, VR14, VR15,
  VR16, VR17, VR18, VR19, VR20, VR21, VR22, VR23,
  VR24, VR25, VR26, VR27, VR28, VR29, VR30, VR31
)>;

// Special register classes
def SpecialReg : RegisterClass<"Tempest", [i64], 64, (add
  TID
)>;