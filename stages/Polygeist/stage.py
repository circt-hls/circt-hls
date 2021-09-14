import shutil
from pathlib import Path
import os

from fud.stages import SourceType, Stage
from fud.utils import shell


class PolygeistStage(Stage):
    def __init__(self, config):
        super().__init__(
            "c",
            "mlir-affine",
            SourceType.Path,
            SourceType.Stream,
            config,
            "Produces mlir files from a C/C++ program"
        )
        self.setup()

    def _define_steps(self, input_file):
        function = self.config["stages", "circt", "toplevel"]
        cmd = " ".join(
            [
              self.config["external-stages", "polygeist", "exec"],
                f"--function={function}",
                "{input_path}"
            ]
        )

        @self.step()
        def compile_with_polygeist(input_path: SourceType.Path) -> SourceType.Stream:
            """
            Compiles an input C program using Polygeist.
            """
            stream = shell(
                cmd.format(input_path=input_path))
            return stream

        # Schedule
        return compile_with_polygeist(input_file)

__STAGES__ = [PolygeistStage]
