# HLSTool
The following document details `hlstool`. `hlstool` is the main driver of CIRCT-HLS, used to compose the various internal and external tools which encompasses the HLS flow.

While the main goal of `hlstool` is to expose commands that allows the synthesis of C programs to HDL in a single, simple call, a secondary goal is to create a tool that helps the development of the toolchain itself.

Some features of the tool are:

* Accepts relative as well as manual paths but executes all commands in absolute paths.
* Each command executed is printed verbatim to the executing terminal. Given this, tool user can inspect the sequence of commands that was executed and easily copy/paste from the terminal, to reproduce any of the commands. All this, without a cryptic shell script in sight!
* Automatically generates `CMakeLists.txt` files for compiling simulator libraries.
* Iteratively `CMake`s a simulator library to achieve the maximum possible Verilator model parallelism.

`hlstool` is also used to drive the CIRCT-HLS regression test suite. As developers, we want to be able to inspect, adjust and rerun regression tests either when debugging failures or implementing new features. To facilitate this, the tool implements a simple checkpointing mechanism.
This tutorial is structured as a sequence of common use-cases of `hlstool`, each building on top of the prior. 

In general, tool arguments are provided at two levels; the general level and the mode level:
~~~~bash
hlstool [general arguments] {mode} [mode arguments]
~~~~


The mode is intended to adjust `hlstool` behaviour to a specific HLS flow. At time of writing, only the `dynamic-polygeist` mode has been implemented.
Since the tool is ever evolving, please reference the `hlstool --help` output for a complete and up-to-date description on the full set of capabilities of the tool.
The following tutorial assumes that you have the `hlstool` available in your path. The tool will be located in `circt-hls/build/bin`.


**Disclaimer:** Since CIRCT, and by extension CIRCT-HLS, is an evolving and actively developed project, tool APIs are highly volatile and subject to change. While it is hoped that the following tutorial will be a useful reference for the foreseeable future, the most surefire way of seeing how the `hlstool` is currently being used is to reference the CIRCT-HLS regression tests.

### Usecase 1: An example kernel

Create a new file `triangle.c` containing the following C program:
~~~~C
int triangle(int n) {
  int sum = 0;
  for (int i = 1; i <= n; i++)
    sum += i;
  return sum;
}
~~~~
as well as a directory for `hlstool` to output to:  
~~~~bash
$ mkdir triangle_out
$ cd triangle_out
~~~~

The `hlstool` may also be provided an optional `--outdir` argument to specify the output (working) directory of the tool.

To convert our `triangle` function to a DHLS Verilog program, we will need to specify:

* A path to the C file to convert
* A name of the function to convert within this kernel. The provided function name is interpreted as the top-level function of the kernel
* The HLS mode of the tool

> ~~~~bash
$ hlstool --kernel_file ../triangle.c --kernel_name triangle dynamic
~~~~

Executing this command, you should now see a `triangle.sv` file containing a SystemVerilog implementation of the kernel. Alongside this, a number of intermediate files is available in the output directory. If you are a developer of `hlstool` or CIRCT-HLS, it is recommended to familiarize yourself with the output of `hlstool` which indicates both the order of as well as the commands that resulted in the generation of each of the intermediate files.

### Usecase 2: Testbenches and cosimulation

Create a testbench file `tst_triangle.c` in the same directory where you created the `triangle.c` file:
~~~~C
int triangle(int);
int main(void) {
  printf("Triangle(%d) = %d\n", 42, triangle(42));
  return 0;
}
~~~~

In testbench mode, `hlstool` is able to infer the kernel name and kernel file based on the assumption that the testbench file is named `tst_{kernel_name}.c` with the kernel file being `{kernel_name}.c` and the kernel name within the kernel file being `{kernel_name}.

Next, we will build the testbench and simulator library. We also pass `--rebuild` to ensure that all steps in the HLS flow are repeated. Internally, `hlstool` has most of its commands guarded by a check on the existence of the output file which it generates. This is used to avoid recompilation in cases where we haven't made any significant changes to the input program, such as when running tests.

~~~~bash
$ hlstool --rebuild --tb_file ../tst_triangle.c dynamic
~~~~

On the testbench side, this will convert the testbench to MLIR using Polygeist (`triangle_affine.mlir`), desynchronize any invocations of the RTL model to `_call/_await` functions (`triangle_tb.mlir`) and lower the testbench to MLIR LLVM (`triangle_tb_llvm.mlir`).

On the kernel side, it is at this point where the simulator library is built. A file `triangle.cpp` should now be present, which is the `hlt` wrapper around the verilated model. A `CMakeLists.txt` file is copied into the output directory. This file contains a call to Verilator, which will verilator the model upon executing CMake. Then, the `hlt` wrapper and the verilated model is compiled and linked together to produces `libhlt_triangle.so` - a shared library which implements the `triangle_call/triangle_await` functions used by the desynchronized.  
Next, we will use `hlstool` to run the simulation:
~~~~bash
$ hlstool --tb_file ../tst_triangle.c dynamic-polygeist --run_sim
~~~~
Inspecting the output of `hlstool`, we see that `mlir-cpu-runner` is invoked to execute the testbench (paths minimized for brevity):
~~~~bash
$ mlir-cpu-runner -e main -entry-point-result=i32 -O3                         \
  -shared-libs=libmlir_c_runner_utils.so -shared-libs=libmlir_runner_utils.so \
  -shared-libs=libhlt_triangle.so triangle_tb_llvm.mlir                       \
  > triangle_tb_output.txt
~~~~
Here we see that `triangle_tb_llvm.mlir` is used as the main MLIR file to execute, and `-shared-libs=libmlir_runner_utils.so` ensures that the simulator library functions are linkable.  
After simulation finishes, three additional files will be available in the output directory:  

* `triangle_tb_output.txt`: `stdout` output generated during execution will be streamed to this file. Within this file, you should be able to see `0`, indicating the return code of the execution, as well as `Triangle(42) = 903`.  
* `logs/vlt_dump.vcd`: VCD output of the verilated model. You can inspect this using tools such as `gtkwave`.  
* `sim.log`: A log printed by the `hlt` infrastructure. This can be used to debug at which steps inputs were pushed and popped to the `hlt` queues that communicates with the transactor interface of the RTL model.

**Note:** In case the Handshake model deadlocks during simulation, an assert will be triggered in the `hlt` infrastructure that is triggered after a fixed number of steps has been performed without any noticeable change in simulator state.  
Going one step further, we may want to cosimulate the RTL simulation with a software implementation of the kernel. Passing `--cosim` will enable cosimulation transformation of the testbench.  
~~~~bash
$ hlstool --rebuild --cosim --tb_file ../tst_triangle.c dynamic
~~~~
Inspecting `triangle_tb.mlir`, we now see calls to the `triangle_call/await` functions and a call to `triangle_ref`. The implementation of `triangle_ref` should have been inlined within the module in `triangle_tb.mlir`.  
The testbench can then be executed as before:  
~~~~bash
$ hlstool --cosim --tb_file ../tst_triangle.c dynamic-polygeist --run_sim
~~~~

Currently, cosimulation failure is indicated through `printf` calls. Given this, failing cases can be identified in the testbench output file `triangle_tb_output.txt`.

### Usecase 3: HSDbg Visualization and Checkpointing

In cases where VCD inspection make be inconclusive in determining the behaviour of a Handshake circuit, HSDbg can be used to provide a visualization of a testbench run. HSDbg can be run within a directory after a simulation has been executed. By default, `hlstool` will look for a VCD file `logs/vlt_dump.vcd`. To point `hlstool` to a custom VCD file, use the `--vcd` argument.
Starting `hsdbg` through `hlstool` is just a small convenience wrapper to ensure that the necessary files are available (Handshake MLIR version of the kernel and a Grapviz `.dot` file of this handshake kernel), and to pass them to `hsdbg`. `hsdbg` is available in `circt-hls/build/bin` and can be invoked separately, if needed.  
In the `triangle_out` directory, run:
~~~~bash
$ hlstool --checkpoint dynamic-polygeist --hsdbg
~~~~

In this commandline, we used the `--checkpoint` option. When the `hlstool` is run, a `.hlstool_checkpoint` file is generated in the working directory of the run. This contains information that can be used to restore the tool arguments at a later point in time. The files loaded from a checkpoint are printed at the start of a run. Executing the above command, `hlstool` will output (path names shortened):
~~~~bash
INFO:     Using input files from .hlstool_checkpoint
INFO:     Using kernel file: ../triangle.c
INFO:     Using kernel name: triangle
INFO:     Using testbench file: ../tst_triangle.c
INFO:     Stored checkpoint to .hlstool_checkpoint. You can rerun HLSTool in
          this directory and pass the --checkpoint flag instead of providing 
          paths and kernel names.
~~~~

On executing the command, a new terminal window will be opened with `hsdbg`, executing and ready to accept commands. `hsdbg` will start a server, defaulting to `localhost:8080` where the visualization will be hosted. Navigate to this page in a browser.  
The webpage will show the current step in the simulation. `hsdbg` does not (yet) have a notion of cycles---the simulation grannularity is identical to that of the timesteps used in the VCD file. In the terminal window executing `hsdbg`, simulation time can be increase by pressing right-arrow, and decreased by pressing left-arrow. A specific step in the simulation can be navigated to by pressing `g` followed by the step number. It may be helpful to use this tool in conjunction with viewing the VCD trace, wherein the VCD trace is used to find situations of interest, and `hsdbg` to get a sense for what is going on at the handshake-level.

### Usecase 4: Modifying kernels at the MLIR level

When developing the HLS flow, we oftentimes want to make precise changes to the MLIR representation of a kerne ---changes, which may not be directly possible when writing the kernel in C. `hlstool` has support for running only a subset of its lowering based on type of the input kernel. In these case, we can provide the `--mlir_kernel` flag.  
For instance, to run the tool starting from the Standard dialect representation of the `triangle` kernel we can run:

~~~~bash
$ cd triangle_out
$ hlstool --rebuild --mlir_kernel --kernel_name triangle \
  --kernel_file triangle_std.mlir dynamic
~~~~

`hlstool` should now inform you that the Polygeist step was skipped:
~~~~bash
INFO:     Skipping Polygeist lowering due to using an MLIR kernel
~~~~
Or with a Handshake IR kernel; in this case we pass the mode argument `--hs_kernel` to direct the Handshake-specific part of `hlstool` to skip lowering:
~~~~bash
$ hlstool --rebuild --mlir_kernel --kernel_name triangle \
  --kernel_file triangle_handshake.mlir dynamic-polygeist --hs_kernel
~~~~

`hlstool` should now inform you that both the Polygeist and handshake steps were skipped:
~~~~bash
INFO:     Skipping Polygeist lowering due to using an MLIR kernel
INFO:     Skipping handshake lowering due to using a handshake kernel
~~~~

**Tip:** After building CIRCT-HLS and running the cosim test suite:
~~~~bash
$ cd build
$ ninja check-circt-hls-cosim
~~~~~

You can easily rerun the `hlstool` invocation by navigating to the output directory of the tests (e.g. `circt-hls/build/cosim_test/suites/Dynamatic/simple_example_1`). This is useful when you want to make slight modifications to the test runs or to reproduce a failing test.

### Usecase 5: Creating a Binary Executable Testbench 

It may happen that we want to write a C/C++ program to directly drive the `call/await` functions of the `hlt` wrapper - this is often the case when developing and debugging the simulation infrastructure itself. When compiled into a regular executable (as opposed to executing testbenches through `mlir-cpu-runner`) This also gives us the ability to easily step through the code while debugging the executable in an IDE.

To illustrate this, we write a testbench `tst_triangle_manual.c` similar to the one shown previously. However, we explicitly reference the call and await functions exposed by the `hlt` transactor interface. Note that we must have predeclarations available for the `call/await` functions. These are available in the `triangle.h` file in the output directory, and generated alongside the `triangle.cpp` file.
 
~~~~C
#include "triangle.h"
int main() {
  triangle_call(42);
  int res = triangle_await();
  printf("Triangle(%d) = %d\n", 42, res);
  return 0;
}
~~~~

The `CMakeLists.txt` file in the output directory contains variables that can be used to create a standalone executable instead of a shared library. To build the above testbench alongside our simulator, we run `cmake` with:

~~~~bash
$ cmake -DHLT_EXEC=1 -DHLT_TESTNAME=triangle -DHLT_EXEC_TB=tst_triangle_manual.c
~~~~

`HLT_EXEC` will trigger building an executable instead of a shared library. By default, a file `main.cpp` is expected to be present in the directory. If this is not the case, you can provide a path to a testbench file with the `-DHLT_EXEC_TB=${path}` variable. The testbench file must contain a `main` function - if not, compilation will fail.  
Then, run `ninja`, and you should have an exectuable `hlt_triangle` in the output directory which you can execute/debug like any other CMake project:
~~~~
$ ninja
$ ./hlt_triangle
Triangle(42) = 903
~~~~
