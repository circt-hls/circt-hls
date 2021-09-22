import shutil
from pathlib import Path
import os

from fud.stages import SourceType, Stage
from fud.utils import shell


class MLIRStageBase(Stage):
    """
    Base stage that defines the common steps between MLIR invocations.
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
            self.config["external-stages", "mlir", "bin_dir"],
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
        def compile_with_MLIR(input_stream: SourceType.Stream) -> SourceType.Stream:
            """
            Compiles an MLIR program using MLIR.
            """
            return shell(cmd, stdin=input_stream)

        # Schedule
        return compile_with_MLIR(input_file)


class MLIRAffineToSCFStage(MLIRStageBase):
    def __init__(self, config):
        super().__init__(
            "mlir-affine",
            "mlir-scf-for",
            "mlir-opt",
            config,
            "Lower MLIR affine to MLIR scf",
            "--lower-affine"
        )

class MLIRSCFForToSCFWhile(MLIRStageBase):
    def __init__(self, config):
        super().__init__(
            "mlir-scf-for",
            "mlir-scf-while",
            "mlir-opt",
            config,
            "Lower MLIR SCF for loops to SCF while loops",
            "--scf-for-to-while"
        )

class MLIRSCFToStandardStage(MLIRStageBase):
    def __init__(self, config):
        super().__init__(
            "mlir-scf-while",
            "mlir-std",
            "mlir-opt",
            config,
            "Lower MLIR SCF to MLIR standard",
            "--convert-scf-to-std"
        )

__STAGES__ = [
    MLIRAffineToSCFStage,
    MLIRSCFToStandardStage,
    MLIRSCFForToSCFWhile
]
