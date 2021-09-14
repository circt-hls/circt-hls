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
                stdin=input_stream,
                stdout_as_debug=True,
            )

        # Schedule
        return compile_with_circt(input_file)


class CIRCTSCFToCalyxStage(CIRCTStageBase):
    def __init__(self, config):
        toplevel = config["stages", "circt", "toplevel"]
        super().__init__(
            "mlir-scf",
            "mlir-calyx",
            "circt-opt",
            config,
            "Lower MLIR SCF to MLIR Calyx",
            f"--lower-scf-to-calyx=top-level-component={toplevel}"
        )

class CIRCTEmitCalyxStage(CIRCTStageBase):
    def __init__(self, config):
        toplevel = config["stages", "circt", "toplevel"]
        super().__init__(
            "mlir-calyx",
            "futil",
            "circt-translate",
            config,
            "Emit MLIR Calyx to native Calyx",
            "--export-calyx"
        )

__STAGES__ = [CIRCTSCFToCalyxStage, CIRCTEmitCalyxStage]
