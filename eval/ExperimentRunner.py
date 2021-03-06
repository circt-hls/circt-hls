#!/usr/bin/env python3
import sys

# Use the report parser already present in Calyx
sys.path.append("../calyx/fud/fud/stages/vivado/")
from rpt import RPTParser

import subprocess
from vcdvcd import VCDVCD
from dataclasses import dataclass
import os
import json
import argparse
import multiprocessing
import concurrent
import re
import shutil
import glob

DYNAMATIC_DIR = ""


def print_yellow(string):
  print("\033[93m{}\033[00m".format(string))


def to_bold(text):
  return "\033[1m{}\033[0m".format(text)


def to_yellow(text):
  return "\033[93m{}\033[00m".format(text)


def to_red(text):
  return "\033[91m{}\033[00m".format(text)


def to_gold_background(text):
  return "\033[42m{}\033[00m".format(text)


def print_header(text, width=80):
  sideWidth = (width - len(text)) // 2 - 2
  text = "=" * sideWidth + f" {text} " + "=" * sideWidth
  print()
  print(to_bold(to_red(to_gold_background(text))))


# print with a


def parse_timing_table(timing_file, table_name):
  """
  Parses the timing table from a timing_summary_routed report generated by vivado.

  This looks like:
  ------------------------------------------------------------------------------------------------
  | Design Timing Summary
  | ---------------------
  ------------------------------------------------------------------------------------------------

      WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints  THS Total Endpoints     WPWS(ns)     TPWS(ns)  TPWS Failing Endpoints  TPWS Total Endpoints  
      -------      -------  ---------------------  -------------------      -------      -------  ---------------------  -------------------     --------     --------  ----------------------  --------------------  
        5.298        0.000                      0                13013        0.038        0.000                      0                13013        4.725        0.000                       0                  5551  
  """
  # find first occurence of "Design timing summary" in timing_file
  with open(timing_file, "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
      if f"| {table_name}" in line:
        break
    if i == len(lines) - 1:
      raise Exception("Could not find Design timing summary in timing_file")
    headerLine = lines[i + 4]
    denomLine = lines[i + 5]
    values = lines[i + 6].strip()

    # split values keeping together things in square brackets
    splitValues = []
    inBracket = False
    bracketValues = []
    for word in values.split():
      if word.startswith("{"):
        inBracket = True
        bracketValues.append(word)
      elif word.endswith("}"):
        inBracket = False
        bracketValues.append(word)
        splitValues.append(" ".join(bracketValues))
        bracketValues.clear()
      elif inBracket:
        bracketValues.append(word)
      else:
        splitValues.append(word)

    denomStartEnds = []
    start = 0
    end = 0
    inLine = False
    for i, c in enumerate(denomLine):
      if c == "-" and not inLine:
        start = i
        inLine = True
      elif c == " " and inLine:
        end = i
        inLine = False
        denomStartEnds.append((start, end))

    headers = []
    for start, end in denomStartEnds:
      headers.append(headerLine[start:end].strip())

    res = {}
    for i, header in enumerate(headers):
      res[header] = splitValues[i]

    return res


def parse_timing_report(timing_file):
  report = {}
  for table in ["Design Timing Summary", "Clock Summary"]:
    report[table] = parse_timing_table(timing_file, table)
  return report


def find_row(table, colname, key, certain=True):
  for row in table:
    if row[colname] == key:
      return row
  if certain:
    raise Exception(f"{key} was not found in column: {colname}")

  return None


def safe_get(d, key):
  if d is not None and key in d:
    return d[key]
  return -1


def to_int(s):
  if s == "-":
    return 0
  return int(s)


class VCDEvaluator:

  def __init__(self, fn):
    print_yellow(
        "Loading VCD trace, this might take a while for large files...")
    self.vcd = VCDVCD(fn)
    self.cycles = None

  def exectime(self, cp):
    """
    Returns the estimated execution time of this VCD file given a clock period.
    """
    cycles = self.get_cycles()
    return cycles * cp

  def getTrace(self, signalcandidates):
    trace = None
    for n in signalcandidates:
      if n in self.vcd.signals:
        trace = self.vcd[n]
        break
    if trace is None:
      raise Exception("Could not find clock signal")
    return trace

  def get_cycles(self):
    """
    Estimates the number of cycles in the vcd file.
    """
    if self.cycles:
      return self.cycles

    clockTrace = self.getTrace(["clock", "clk", "TOP.clock", "TOP.clk"])
    resetTrace = self.getTrace(["reset", "rst", "TOP.reset", "TOP.rst"])

    if len(resetTrace.tv) != 2:
      raise Exception("Expected two values in the reset signal vector")

    # [1] gives us the value after the init value of the reset signal
    # [0] indexes into the (step, value) tuple
    outOfReset = resetTrace.tv[1][0]

    OORClockTrace = None

    # Find the first index in the clock trace after the reset signal has flipped
    for i in range(len(clockTrace.tv)):
      if clockTrace.tv[i][0] > outOfReset:
        OORClockTrace = clockTrace.tv[i:]
        break

    # number of cycles is equal to the number of entries in the clock trace divided
    # by two (high, followed by low).
    self.cycles = len(OORClockTrace) // 2
    return self.get_cycles()


def run_hls_tool(args):
  cmd = " ".join(args)
  print_yellow("Running HLSTool with: {}".format(cmd))
  proc = subprocess.Popen(cmd,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          shell=True)
  # Continuously poll the process and forward any stdout
  # if proc is finished
  while not proc.poll():
    stdoutdata = proc.stdout.readline()
    if stdoutdata:
      sys.stdout.write(stdoutdata.decode("utf-8"))
      sys.stdout.flush()
    else:
      break

  proc.wait()
  stdErr = proc.stderr.read().decode("utf-8")
  code = proc.returncode

  # if code and code != 0 or stdErr:
  #   print(stdErr)
  #   sys.exit(1)


class HLTLogEval:

  def __init__(self, log_file):
    self.log_file = log_file
    self.cycles = None

  def exectime(self, cp):
    """
    Returns the estimated execution time of this VCD file given a clock period.
    """
    cycles = self.get_cycles()
    return cycles * cp

  def get_cycles(self):
    """ The HLT log file has lines on the format
    {clock cycle} @ {info message}

    We assume that the number of cycles executed was the number of cycles for the
    last "OUT TO WAITER" message.
  """
    if self.cycles:
      return self.cycles

    with open(self.log_file, "r") as f:
      lines = f.readlines()
      for line in reversed(lines):
        if "OUT TO WAITER" in line:
          self.cycles = int(line.split("@")[0].strip())
          return self.get_cycles()

    raise Exception("Could not find number of cycles in HLT log file")


@dataclass
class Experiment:
  # Name of the experiment
  experimentName: str
  # Name of this experiment run
  name: str
  # Testbench file of the experiment. The kernel name is inferred from this.
  tb: str
  # Number of times the kernel is called (in a loop) in its testbench
  kernel_calls: int
  # The HLS mode
  mode: str
  # Generic arguments (these come before the 'mode' in the hlstool commandline)
  args: list
  # mode arguments
  mode_args: list
  # Run synthesis
  synth: bool = True
  # Run simulation
  sim: bool = True
  # dynamatic or circt-hls
  style: str = "default"

  def run(self):
    print_header("Running experiment: " + self.name)

    self.outdir = os.path.join(os.getcwd(), "results", self.experimentName,
                               self.name)
    if not os.path.exists(self.outdir):
      os.makedirs(self.outdir)

    if self.style == "circt-hls":
      self.run_hlstool()
    elif self.style == "dynamatic":
      self.run_dynamatic()

    if self.sim:
      # Extract # of cycles executed from simulator log file.
      simlogpath = os.path.join(self.outdir, "sim.log")
      self.cycleeval = HLTLogEval(simlogpath)
      print_yellow(
          f"Estimated execution time: {self.cycleeval.get_cycles()} cycles")
    if self.synth:
      # Get reports
      rpts = []
      for root, dirs, files in os.walk(self.outdir):
        for f in files:
          if f.endswith(".rpt"):
            rpts.append(os.path.join(root, f))

      def getReport(name):
        for rpt in rpts:
          if name in rpt:
            return rpt
        return None

      utilReport = RPTParser(getReport("utilization_placed"))
      timingFile = getReport("timing_summary_routed")
      self.print_summary(utilReport, timingFile)

  def print_summary(self, util, timingFile):
    slice_logic = util.get_table(re.compile(r"1\. CLB Logic"), 2)
    CLB_logic = util.get_table(re.compile(r"2\. CLB Logic Distribution"), 2)
    dsp_table = util.get_table(re.compile(r"4\. ARITHMETIC"), 2)
    timing_report = parse_timing_report(timingFile)

    summary_file = os.path.join(self.outdir, "experiment_summary.txt")
    with open(summary_file, "w") as f:
      f.write("# Experiment setup:\n")
      f.write("   TB: " + self.tb + "\n")
      f.write("   Kernel calls: " + str(self.kernel_calls) + "\n")
      f.write("   Mode: " + self.mode + "\n")
      f.write("   Mode args: " + str(self.mode_args) + "\n")
      f.write("   Args: " + str(self.args) + "\n")
      f.write("\n")

      f.write("# Experiment results:\n")
      wsn = float(timing_report["Design Timing Summary"]["WNS(ns)"])
      f.write("WNS(ns): " + str(wsn) + "\n")
      cp = float(timing_report["Clock Summary"]["Period(ns)"])
      f.write("Clock period(ns): " + str(cp) + "\n")
      max_cp = cp - wsn
      f.write("Max clock period(ns): " + str(max_cp) + "\n")

      if self.sim:
        min_exectime = float(self.cycleeval.exectime(max_cp))
        exectime = float(self.cycleeval.exectime(cp))
        f.write("cycles executed: " + str(self.cycleeval.get_cycles()) + "\n")
        f.write("Execution time(ns): " + str(exectime) + "\n")
        f.write("Min execution time(ns): " + str(min_exectime))

      clb = to_int(find_row(CLB_logic, "Site Type", "CLB")["Used"])
      clb_lut = to_int(find_row(slice_logic, "Site Type", "CLB LUTs")["Used"])
      clb_reg = to_int(
          find_row(slice_logic, "Site Type", "CLB Registers")["Used"])
      dsp = to_int(find_row(dsp_table, "Site Type", "DSPs")["Used"])

      f.write("\n")
      f.write("CLB logic:\n")
      f.write("   CLBs logic: " + str(clb) + "\n")
      f.write("   CLB LUTs: " + str(clb_lut) + "\n")
      f.write("   CLB Registers: " + str(clb_reg) + "\n")
      f.write("   DSPs: " + str(dsp) + "\n")

    print_yellow(f"Created summary for experiment {self.name} at: " +
                 summary_file)
    with open(summary_file, "r") as f:
      print(f.read())

  def run_hlstool(self):
    hlstool_args = ["hlstool"]
    if self.synth:
      hlstool_args.append("--synth")  # Synthesize the design
    hlstool_args.append("--rebuild")  # Always ensure fresh results
    hlstool_args.append("--tb_file " + self.tb)
    hlstool_args.append("--outdir " + self.outdir)
    hlstool_args += self.args
    # The # of kernel calls for this test is controlled through the #define in
    # the testbenches. This is elaborated in the polygeist front-end, so inject
    # that as an argument.
    hlstool_args.append(
        f"--extra_polygeist_tb_args=\"-DN_KERNEL_CALLS={self.kernel_calls}\"")
    hlstool_args.append(self.mode)
    if self.sim:
      hlstool_args.append("--run_sim")  # Run the testbench
    hlstool_args += self.mode_args
    run_hls_tool(hlstool_args)

  def run_dynamatic(self):
    # Dynamatic expects the kernel to be within a "src" directory. It is ok that the dir exists
    # since it is created by the front-end.
    src_dir = os.path.join(self.outdir, "src")
    os.makedirs(src_dir, exist_ok=True)

    # Copy the .sv file in the 'hdl' directory
    sv_files = glob.glob(os.path.join(self.tb, "hdl", "*.vhd"))
    # expect at least 1 sv file
    assert len(sv_files) > 0
    for sv_file in sv_files:
      shutil.copy(sv_file, self.outdir)

    # Copy the dynamatic library files to the hdl directory
    dynamatic_lib_files = glob.glob(
        os.path.join(DYNAMATIC_DIR, "components", "*.vhd"))
    for lib_file in dynamatic_lib_files:
      shutil.copy(lib_file, os.path.join(self.outdir))

    # Copy synthesis files to the output directory.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    synth_tcl = os.path.join(script_dir, os.path.pardir, "tools", "hlstool",
                             "synth.tcl")
    device_xdc = os.path.join(DYNAMATIC_DIR, "device.xdc")
    shutil.copy(synth_tcl, self.outdir)
    shutil.copy(device_xdc, self.outdir)

    curdir = os.getcwd()
    os.chdir(self.outdir)
    # Run vivado
    vivado_args = ["vivado", "-mode", "batch", "-source", "synth.tcl"]
    # synth arguments (see synth.tcl)
    vivado_args.append("-tclargs")
    vivado_args.append(self.name)  # top level
    vivado_args.append("xczu3eg-sbva484-1-e")  # part
    vivado_args.append("vivado")  # outdir
    vivado_args.append("1")  # do routing
    subprocess.run([" ".join(vivado_args)], shell=True)
    os.chdir(curdir)


# =============================================================================
# Experiments
# =============================================================================

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Run a set of experiments on a given testbench.")
  parser.add_argument("experiments",
                      help="JSON file containing the experiments to run.",
                      type=str)
  parser.add_argument(
      "--concurrency",
      help="Run the experiments concurrently (specify # of threads)",
      type=int,
      default=1)

  parser.add_argument("--dynamatic_dir",
                      help="Path to the dynamatic directory",
                      type=str)

  args = parser.parse_args()

  DYNAMATIC_DIR = args.dynamatic_dir

  if not os.path.isfile(args.experiments):
    print("Experiments file does not exist: ", args.experiments)
    exit(1)

  # get path without extension
  # Get basename of args.experiments
  experiments_file = os.path.basename(args.experiments)
  experiments_file = os.path.splitext(experiments_file)[0]

  experiments = []
  with open(args.experiments) as f:
    expFile = json.load(f)

    # Load the test setup
    setup = expFile["setup"]
    mode = setup["mode"]
    general_args = setup["args"]
    mode_args = setup["mode_args"]
    kernel_calls = setup["kernel_calls"]
    synth = setup["synth"]
    sim = setup["sim"]
    style = setup["style"]

    def overrideIfExists(container, key, current):
      if key in container:
        return container[key]
      return current

    # Load the experiment runs
    for runName, expValues in expFile["runs"].items():
      # Experiment runs are allowed to override the default setup specified in
      # the experiment file.
      run_mode = overrideIfExists(expValues, "mode", mode)
      run_general_args = overrideIfExists(expValues, "args", general_args)
      run_mode_args = overrideIfExists(expValues, "mode_args", mode_args)
      run_kernel_calls = overrideIfExists(expValues, "kernel_calls",
                                          kernel_calls)
      run_synth = overrideIfExists(expValues, "synth", synth)
      run_sim = overrideIfExists(expValues, "sim", sim)
      run_tb = expValues["tb"]

      experiments.append(
          Experiment(experimentName=experiments_file,
                     name=runName,
                     tb=run_tb,
                     kernel_calls=run_kernel_calls,
                     mode=run_mode,
                     args=run_general_args,
                     mode_args=run_mode_args,
                     synth=run_synth,
                     sim=run_sim,
                     style=style))

  print_yellow("Loaded experiments:")
  for experiment in experiments:
    print_yellow("  " + experiment.name)

  if not os.path.exists("results"):
    os.makedirs("results")

  if not os.path.exists(os.path.join("results", experiments_file)):
    os.makedirs(os.path.join("results", experiments_file))

  # Run the experiments
  from concurrent.futures import ThreadPoolExecutor
  futures = []
  with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
    for experiment in experiments:
      futures.append(executor.submit(experiment.run))
    # join
    for future in concurrent.futures.as_completed(futures):
      future.result()
