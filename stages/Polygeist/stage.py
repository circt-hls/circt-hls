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

    @staticmethod
    def defaults():
        return {
            "exec": "mlir-clang"
        }

    def _define_steps(self, input_file):
        function = self.config["stages", "circt_hls", "toplevel"]
        cmd = " ".join(
            [
                # Arguments are:
                # --function= all functions in file
                # --S = emit assembly
                # --memref-fullrank = emit fullrank memrefs (non-dynamic memref sizes).
              self.config["stages", "polygeist", "exec"],
                f"--function=* -S -memref-fullrank",
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
