import subprocess
import os
from pathlib import Path

OUTDIR = None


def print_header(string, width=80):
    n = int((width - len(string) - 2) / 2)
    print("\n" + n * "=" + f" {string} " + n * "=")


def run_fud(fn, argFrom, argTo, flags=""):
    subprocess.run(["fud", "exec", "--from", argFrom,
                    "--to", argTo, flags, fn])


def run_vivado(path):
    print_header("Vivado run")
    # run_fud(path + ".c", "vivado-hls", "synth-files", "-o " + str(OUTDIR))


def run_dynamic(path):
    print_header("Dynamic run")
    dynamic_dir = OUTDIR / "dynamic"
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    tmpfile = dynamic_dir / os.path.basename(path)
    run_fud(path / "main.mlir", "mlir-affine", "mlir-handshake",
            "-o" + str(tmpfile) + "_handshake.mlir")
    #run_fud(tmpfile, "mlir-handshake", "synth-files", "-o " + str(dynamic_dir))


def run_static(path):
    print_header("Static run")
    static_dir = OUTDIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    tmpfile = static_dir / os.path.basename(path)
    run_fud(path / "main.mlir", "mlir-affine",
            "futil", "-o" + str(tmpfile) + ".futil")
    #run_fud(tmpfile, "futil", "synth-files", "-o " + str(static_dir))


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
        "--dynamic", help="Executes the dynamically scheduled flow", action="store_true")
    parser.add_argument(
        "--static", help="Executes the statically scheduled flow", action="store_true")

    args = parser.parse_args()
    dir_base = os.path.basename(args.dir)
    OUTDIR = Path.cwd() / "out" / "comparisons" / dir_base
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dir_path = Path(args.dir)

    if args.vivado_hls:
        run_vivado(dir_path)

    if args.dynamic:
        run_dynamic(dir_path)

    if args.static:
        run_static(dir_path)
