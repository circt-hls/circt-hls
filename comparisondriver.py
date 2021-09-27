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

    def run(self):
        [runner.run() for runner in self.runs]

    def getFilename(self, suffix):
        tmpfile = self.dir / suffix
        return str(tmpfile)

    def getResourcesFile(self):
        return self.getFilename("resources.json")

    def getSynthResourceEstimateRun(self, synthDir):
        return FUDRun(
            str(self.dir / synthDir),
            "synth-files",
            "resource-estimate",
            [f"-o{self.getResourcesFile()}"]
        )

    def getInlineRun(self, inlineDir, extension, toplevel):
        """ Inlines all files in inlineDir with the provided extension into the
        top-level file.
        """
        def inlineFunc():
            toInline = ""
            for srcfile in os.scandir(inlineDir):
                if srcfile.name == toplevel:
                    continue

                if srcfile.name.endswith(f".{extension}"):
                    with open(srcfile) as f:
                        toInline = toInline + f.read() + "\n"

            with open(inlineDir / toplevel, "r+") as topFile:
                toInline += topFile.read()
                topFile.seek(0, 0)
                topFile.write(toInline)

        return CustomRun(inlineFunc)

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
            shutil.move(tmpdir, self.dir / renameTo)

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
            name="vivado-hls",
            testPath=path
        )
        # Generate HLS files
        self.runs.append(
            FUDRun(path / "main.c", "vivado-hls", "hls-files",
                   [f"-o{self.dir}"]))
        # Rename the temporary hls folder
        self.runs.append(self.getTmpFolderRenameRun("hls_output"))

        # Since fud only accepts a single verilog file, inline any additional
        # generated verilog files into the top-level, and hope for the best!
        hls_verilogDir = self.dir / "hls_output" / \
            "benchmark.prj" / "solution1" / "syn" / "verilog"
        self.runs.append(self.getInlineRun(hls_verilogDir, "v", "main.v"))

        # Synthesize the top-level verilog file
        self.runs.append(
            FUDRun(hls_verilogDir / "main.v",
                   "synth-verilog", "synth-files", [f"-o{self.dir}"]))

        # Rename temporary folder, run synthesis, and get resource estimate
        self.runs.append(self.getTmpFolderRenameRun("vivado"))
        self.runs.append(self.getSynthResourceEstimateRun("vivado"))


class DynamicRunner(FUDRunner):
    def __init__(self, path):
        super().__init__(
            name="circt-dynamic",
            testPath=path
        )
        # Run affine to handshake
        handshake_tmpfile = self.getFilename(f"{self.testName}_handshake.mlir")
        self.runs.append(
            FUDRun(path / "main.mlir", "mlir-affine", "mlir-handshake",
                   [f"-o{handshake_tmpfile}"])
        )
        # Run handshake to synth-files; this goes through FIRRTL -> verilog
        self.runs.append(
            FUDRun(handshake_tmpfile, "mlir-handshake",
                   "synth-files", [f"-o{self.dir}"])
        )

        # Rename temporary folder, run synthesis, and get resource estimate
        self.runs.append(self.getTmpFolderRenameRun("vivado"))
        self.runs.append(self.getSynthResourceEstimateRun("vivado"))


class StaticRunner(FUDRunner):
    def __init__(self, path):
        super().__init__(
            name="circt-static",
            testPath=path
        )

        # Run affine to futil
        futil_tmpfile = self.getFilename(f"{self.testName}.futil")
        self.runs.append(
            FUDRun(path / "main.mlir", "mlir-affine",
                   "futil", [f"-o{futil_tmpfile}"])
        )

        # Run futil to synth files; this goes through the native calyx compiler
        self.runs.append(
            FUDRun(futil_tmpfile, "futil", "synth-files", [f"-o{self.dir}"])
        )

        # Rename temporary folder, run synthesis, and get resource estimate
        self.runs.append(self.getTmpFolderRenameRun("vivado"))
        self.runs.append(self.getSynthResourceEstimateRun("vivado"))


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
