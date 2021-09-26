import subprocess
import os
from pathlib import Path

OUTDIR = None


def print_header(string, width=80):
    n = int((width - len(string) - 2) / 2)
    print("\n" + n * "=" + f" {string} " + n * "=")


def run_fud(fn, argFrom, argTo, flags=""):
    cmd = ["fud", "exec", "--from", argFrom,
           "--to", argTo, flags, str(fn), "--verbose"]
    print("Running fud with: " + " ".join(cmd))
    subprocess.run(cmd)


class FUDRunner():
    def __init__(self, testPath, name):
        print_header(f"{name} run")
        self.dir = OUTDIR / name
        newDir = self.dir
        i = 1
        while newDir.exists():
            newDir = Path(str(self.dir) + f"_{i}")
            i += 1
        self.dir = newDir

        self.dir.mkdir(parents=True, exist_ok=True)
        self.fudRuns = []
        self.testPath = testPath

    def run(self):
        for run in self.fudRuns:
            run_fud(*run)

    def getTempFile(self, suffix):
        tmpfile = self.dir / os.path.basename(self.testPath)
        return str(tmpfile) + suffix


class VivadoHLSRunner(FUDRunner):
    def __init__(self, path):
        super().__init__(
            name="Vivado HLS",
            testPath=path
        )
        self.fudRuns.append(
            [path / ".c", "vivado-hls", "synth-files",
                "-o" + str(self.dir)]
        )


class DynamicRunner(FUDRunner):
    def __init__(self, path):
        super().__init__(
            name="Dynamic",
            testPath=path
        )
        handshake_tmpfile = self.getTempFile("_handshake.mlir")
        self.fudRuns.append(
            [path / "main.mlir", "mlir-affine", "mlir-handshake",
                "-o" + handshake_tmpfile]
        )

        self.fudRuns.append(
            [handshake_tmpfile, "mlir-handshake",
                "synth-files", "-o " + str(self.dir)]
        )


class StaticRunner(FUDRunner):
    def __init__(self, path):
        super().__init__(
            name="Static",
            testPath=path
        )
        futil_tmpfile = self.getTempFile(".futil")
        self.fudRuns.append(
            [path / "main.mlir", "mlir-affine", "futil", "-o" + futil_tmpfile]
        )
        self.fudRuns.append(
            [futil_tmpfile, "futil", "synth-files", "-o " + str(self.dir)]
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CIRCT HLS comparison driver"
    )

    parser.add_argument(
        "dir", type=str, help="Input directory.\n"
        "  For Vivado a main.c file must be present\n"
        "  For the CIRCT flows, a main.mlir file must be present.")
    parser.add_argument(
        "--vivado-hls", help="Executes Vivado HLS", action="store_true")
    parser.add_argument(
        "--circt-dynamic", help="Executes the CIRCT-based dynamically scheduled flow", action="store_true")
    parser.add_argument(
        "--circt-static", help="Executes the CIRCT-based statically scheduled flow", action="store_true")

    args = parser.parse_args()
    dir_base = os.path.basename(args.dir)
    OUTDIR = Path.cwd() / "out" / "comparisons" / dir_base
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dir_path = Path(args.dir)

    if args.vivado_hls:
        VivadoHLSRunner(dir_path).run()

    if args.circt_dynamic:
        DynamicRunner(dir_path).run()

    if args.circt_static:
        StaticRunner(dir_path).run()
