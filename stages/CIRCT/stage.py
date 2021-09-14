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
        self.flags = flags
        self.setup()

    def _define_steps(self, input_file):
        cmd = " ".join(
            [
                self.config["external-stages", "circt", "exec"],
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
        super().__init__(
            "mlir-scf",
            "mlir-calyx",
            config,
            "Lower MLIR SCF to MLIR Calyx",
            "--lower-scf-to-calyx"
        )


__STAGES__ = [CIRCTSCFToCalyxStage]
