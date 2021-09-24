import shutil
from pathlib import Path
import os

from fud.stages import SourceType, Stage
from fud.utils import shell


class CIRCTStageBase(Stage):
    """
    Base stage that defines the common steps between CIRCT invocations.
    """

    def __init__(
        self,
        src,
        dst,
        executable,
        config,
        description,
        flags
    ):
        super().__init__(
            src,
            dst,
            SourceType.Stream,
            SourceType.Stream,
            config,
            description
        )
        self.executable = os.path.join(
            self.config["external-stages", "circt", "bin_dir"],
            executable)
        self.flags = flags
        self.setup()

    def _define_steps(self, input_file):
        cmd = " ".join(
            [
                self.executable,
                self.flags
            ]
        )

        @self.step()
        def compile_with_circt(input_stream: SourceType.Stream) -> SourceType.Stream:
            """
            Compiles an MLIR program using CIRCT.
            """
            return shell(
                cmd,
                stdin=input_stream
            )

        # Schedule
        return compile_with_circt(input_file)


class CIRCTSCFToDataflow(CIRCTStageBase):
    def __init__(self, config):
        super().__init__(
            "mlir-std",
            "mlir-handshake",
            "circt-opt",
            config,
            "Lower MLIR SCF to MLIR Handshake dialect",
            "-create-dataflow" # | circt-opt -canonicalize-dataflow | circt-opt -remove-block-structure | circt-opt -handshake-insert-buffer"
        )

class CIRCTHandshakeToFIRRTL(CIRCTStageBase):
    def __init__(self, config):
        super().__init__(
            "mlir-handshake",
            "mlir-firrtl",
            "circt-opt",
            config,
            "Lower MLIR Handshake to MLIR FIRRTL",
            "-lower-handshake-to-firrtl"
        )

class CIRCTFIRRTLToHW(CIRCTStageBase):
    def __init__(self, config):
        super().__init__(
            "mlir-firrtl",
            "mlir-hw",
            "circt-opt",
            config,
            "Lower FIRRTL types to ground types",
            "-pass-pipeline='module(firrtl.circuit(firrtl-lower-types), " +
            "firrtl.circuit(firrtl-infer-widths), "+ 
            "builtin.module(lower-firrtl-to-hw))'"
        )

class CIRCTSCFToCalyxStage(CIRCTStageBase):
    def __init__(self, config):
        toplevel = config["stages", "circt_hls", "toplevel"]
        super().__init__(
            "mlir-scf-while",
            "mlir-calyx",
            "circt-opt",
            config,
            "Lower MLIR SCF to MLIR Calyx",
            f"--lower-scf-to-calyx=top-level-component={toplevel}"
        )

class CIRCTEmitCalyxStage(CIRCTStageBase):
    def __init__(self, config):
        super().__init__(
            "mlir-calyx",
            "futil",
            "circt-translate",
            config,
            "Emit MLIR Calyx to native Calyx",
            "--export-calyx"
        )

class CIRCTFIRRTLToVerilog(CIRCTStageBase):
    def __init__(self, config):
        super().__init__(
            "mlir-firrtl",
            "synth-verilog",
            "firtool",
            config,
            "Convert FIRRTL to verilog",
            "--verilog --format=mlir"
        )

__STAGES__ = [
    CIRCTSCFToCalyxStage,
    CIRCTEmitCalyxStage,
    CIRCTHandshakeToFIRRTL,
    CIRCTSCFToDataflow,
    CIRCTFIRRTLToVerilog
]
