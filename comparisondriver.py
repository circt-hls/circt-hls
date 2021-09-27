import subprocess
import os
from pathlib import Path
import sys
import shutil
from dataclasses import dataclass, field

OUTDIR = None


def print_header(string, width=80):
    n = int((width - len(string) - 2) / 2)
    print("\n" + n * "=" + f" {string} " + n * "=")


@dataclass
class FUDRun:
    filename: str
    argFrom: str
    argTo: str
    flags: list = field(default_factory=list)

    def run(self):
        cmd = ["fud", "exec", "--from", self.argFrom,
               "--to", self.argTo, " ".join(self.flags), str(
                   self.filename), "--verbose"
               ]
        print("Running fud with: " + " ".join(cmd))
        subprocess.run(cmd)


@dataclass
class CustomRun:
    func: callable

    def run(self):
        self.func()


class FUDRunner():
    def __init__(self, testPath, name):
        print_header(f"{name} run")
        self.dir = OUTDIR / name
        if self.dir.exists():
            shutil.rmtree(self.dir)

        self.dir.mkdir(parents=True, exist_ok=True)
        self.runs = []
        self.testPath = testPath
        self.testName = os.path.basename(testPath)
        self.vivadoDir = self.dir / "vivado"

    def run(self):
        [runner.run() for runner in self.runs]

    def getFilename(self, suffix):
        tmpfile = self.dir / suffix
        return str(tmpfile)

    def getResourceEstimateRun(self):
        resourcesFile = self.getFilename("resources.json")
        return FUDRun(
            str(self.vivadoDir),
            "synth-files",
            "resource-estimate",
            [f"-o{resourcesFile}"]
        )

    def getTmpFolderRenameRun(self, renameTo):
        """Renames any temporary folder within the output directory to
        """
        def runFunc():
            tmpdir = [x.path for x in os.scandir(
                self.dir) if x.is_dir() and x.name.startswith("tmp")]
            if len(tmpdir) != 1:
                print("ERROR: Expected a single temporary directory")
                sys.exit(1)
            tmpdir = tmpdir[0]
            shutil.move(tmpdir, renameTo)

        return CustomRun(runFunc)

    def run_fud(self, fn, argFrom, argTo, flags=[]):
        cmd = ["fud", "exec", "--from", argFrom,
               "--to", argTo, " ".join(flags), str(fn), "--verbose"
               ]
        print("Running fud with: " + " ".join(cmd))
        subprocess.run(cmd)


class VivadoHLSRunner(FUDRunner):
    def __init__(self, path):
        super().__init__(
            name="Vivado HLS",
            testPath=path
        )
        self.runs.append(
            FUDRun(path / ".c", "vivado-hls", "synth-files",
                   [f"-o{self.dir}"])
        )
        self.runs.append(self.getTmpFolderRenameRun(self.vivadoDir))
        self.runs.append(self.getResourceEstimateRun())


class DynamicRunner(FUDRunner):
    def __init__(self, path):
        super().__init__(
            name="Dynamic",
            testPath=path
        )
        handshake_tmpfile = self.getFilename(f"{self.testName}_handshake.mlir")
        self.runs.append(
            FUDRun(path / "main.mlir", "mlir-affine", "mlir-handshake",
                   [f"-o{handshake_tmpfile}"])
        )
        self.runs.append(
            FUDRun(handshake_tmpfile, "mlir-handshake",
                   "synth-files", [f"-o{self.dir}"])
        )
        self.runs.append(self.getTmpFolderRenameRun(self.vivadoDir))
        self.runs.append(self.getResourceEstimateRun())


class StaticRunner(FUDRunner):
    def __init__(self, path):
        super().__init__(
            name="Static",
            testPath=path
        )
        futil_tmpfile = self.getFilename(f"{self.testName}.futil")
        self.runs.append(
            FUDRun(path / "main.mlir", "mlir-affine",
                   "futil", [f"-o{futil_tmpfile}"])
        )
        self.runs.append(
            FUDRun(futil_tmpfile, "futil", "synth-files", [f"-o{self.dir}"])
        )
        self.runs.append(self.getTmpFolderRenameRun(self.vivadoDir))
        self.runs.append(self.getResourceEstimateRun())


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
