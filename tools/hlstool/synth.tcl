# HLSTool general synthesis script

if { $argc != 4 } {
  puts "The synth.tcl script requires three arguments"
  puts "Usage: synth <top level> <part name> <output directory> <do route>"
  exit 1
}
set top [lindex $argv 0]
set part [lindex $argv 1]
set outdir [lindex $argv 2]
set doRouting [lindex $argv 3]

set_param general.maxThreads 8

# Create the project (forcibly overwriting) and add sources SystemVerilog
# (*.sv) and Xilinx constraint files (*.xdc), which contain directives for
# connecting design signals to physical FPGA pins.
create_project -force -part $part $top $outdir
add_files [glob ./*.sv]
add_files -fileset constrs_1 [glob ./*.xdc]
set_property top $top [current_fileset]

# Switch the project to "out-of-context" mode, which frees us from the need to
# hook up every input & output wire to a physical device pin.
set_property \
    -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
    -value {-mode out_of_context -flatten_hierarchy "rebuilt"} \
    -objects [get_runs synth_1]

# Run synthesis. This is enough to generate the utilization report mentioned
# above but does not include timing information.
launch_runs synth_1
wait_on_run synth_1

if { $doRouting != 0} {
  # Run implementation to do place & route. This also produces the timing
  # report mentioned above. Removing this step makes things go quite a bit
  # faster if you just need the resource report!
  launch_runs impl_1 -to_step route_design
  wait_on_run impl_1
}