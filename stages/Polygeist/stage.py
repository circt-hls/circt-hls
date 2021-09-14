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


class LowerAffineStage(Stage):
    def __init__(self, config):
        super().__init__(
            "mlir-affine",
            "mlir-scf",
            SourceType.Stream,
            SourceType.Stream,
            config,
            "Lowers a polygeist output to a point suitable for CIRCT ingress"
        )
        self.executable = os.path.join(
            self.config["external-stages", "circt", "llvm_bin_dir"], "mlir-opt")
        self.setup()

    def _define_steps(self, input_stream):
        cmd = " ".join([self.executable, "--lower-affine"])

        @self.step()
        def compile_with_mlir(input_stream: SourceType.Stream) -> SourceType.Stream:
            """
            Lowers an input program using MLIR.
            """
            stream = shell(cmd, stdin=input_stream)
            return stream

        # Schedule
        return compile_with_mlir(input_stream)


__STAGES__ = [PolygeistStage, LowerAffineStage]
