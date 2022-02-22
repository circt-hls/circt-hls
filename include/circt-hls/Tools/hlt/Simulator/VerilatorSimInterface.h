#ifndef CIRCT_TOOLS_HLT_VERILATORSIMINTERFACE_H
#define CIRCT_TOOLS_HLT_VERILATORSIMINTERFACE_H

#include <functional>

#include "circt-hls/Tools/hlt/Simulator/SimDriver.h"

#include "verilated.h"

#if VM_TRACE
#include "verilated_vcd_c.h"
#endif

// Legacy function required only so linking works on Cygwin and MSVC++
double sc_time_stamp() { return 0; }

namespace circt {
namespace hlt {

template <typename TSigType>
class VerilatorSignal {
  // This class encapsulates a top-level verilator signal. We override the
  // assignment operator to be able to track modifications to the top-level
  // I/O of a verilated model.
public:
  VerilatorSignal(TSigType *sig) : m_sig(sig) {}

  VerilatorSignal &operator=(const TSigType &rhs) {
    assert(m_sig && "VerilatorSignal: null signal");
    *m_sig = rhs;
    return *this;
  }

  operator TSigType() const {
    assert(m_sig && "VerilatorSignal: null signal");
    return *m_sig;
  }

  /// Assigns a value to this verilator signal, and returns true if the
  /// assignment modified the current value of the signal.
  bool assign(const TSigType &rhs) {
    assert(m_sig && "VerilatorSignal: null signal");
    if (*m_sig != rhs) {
      *m_sig = rhs;
      return true;
    }
    return false;
  }

private:
  /// Pointer to a member variable of a verilated model that represents an I/O
  /// signal.
  TSigType *m_sig = nullptr;
};

/// Generic interface to access various parts of the verilated model. This is
/// needed due to verilator models themselves not inheriting from some form of
/// interface.
struct VerilatorGenericInterface {
  CData *clock = nullptr;
  CData *reset = nullptr;
  CData *nReset = nullptr;
};

template <typename TInput, typename TOutput, typename TModel>
class VerilatorSimInterface : public SimInterface<TInput, TOutput> {
public:
  VerilatorSimInterface() : SimInterface<TInput, TOutput>() {
    // Instantiate the verilated model
    ctx = std::make_unique<VerilatedContext>();
    dut = std::make_unique<TModel>(ctx.get());

#if VM_TRACE
    ctx->traceEverOn(true);
    trace = std::make_unique<VerilatedVcdC>();
    // Log 99 levels of hierarchy
    dut->trace(trace.get(), 99);
    // Create logging output directory
    Verilated::mkdir("logs");
    trace->open("logs/vlt_dump.vcd");
#endif
  }

  uint64_t time() override {
    // We could return ctx->time() here, but a more accurate representation
    // would be the number of cycles executed.
    return m_clockCycles;
  }

  void step() override {
    clock_flip();
    m_clockCycles++;
  }

  void dump(std::ostream &out) const override {
    out << "Port states:\n";
    for (auto &inPort : this->inPorts)
      out << *inPort << "\n";
    for (auto &outPort : this->outPorts)
      out << *outPort << "\n";
    out << "\n";
  }

  void setup() override {
    // Verify generic interface
    assert(interface.clock != nullptr && "Must set pointer to clock signal");
    assert((static_cast<bool>(interface.reset) ^
            static_cast<bool>(interface.nReset)) &&
           "Must set pointer to either reset or nReset");

    // Reset top-level model
    if (interface.reset)
      *interface.reset = !0;
    else
      *interface.nReset = !1;

    // Reset in- and output ports
    for (auto &port : this->inPorts)
      port->reset();
    for (auto &port : this->outPorts)
      port->reset();

    // Run for a few cycles with reset.
    for (int i = 0; i < 2; ++i)
      this->clock();

    // Disassert reset
    if (interface.reset)
      *interface.reset = !1;
    else
      *interface.nReset = !0;
    this->clock();
  }

  void finish() override {
    dut->final();

#if VM_TRACE
    // Close trace if opened.
    trace->close();
#endif
  }

protected:
  void advanceTime() {
#if VM_TRACE
    trace->dump(ctx->time());
    // If tracing, flush after each cycle so we can immediately see the output.
    trace->flush();
#endif
    ctx->timeInc(1);
    dut->eval();
  }

  // Clocks the model a half phase (rising or falling edge)
  void clock_half(bool rising) {
    // Ensure combinational logic is settled, if input pins changed.
    advanceTime();
    *interface.clock = rising;
    dut->eval();
    advanceTime();
  }

  void clock_rising() { clock_half(true); }
  void clock_falling() { clock_half(false); }
  void clock_flip() { clock_half(!*interface.clock); }
  void clock() {
    clock_rising();
    clock_falling();
  }

  // Pointer to the verilated model.
  std::unique_ptr<TModel> dut;
  std::unique_ptr<VerilatedContext> ctx;
  VerilatorGenericInterface interface;

  // Number of clock-cycles executed.
  uint64_t m_clockCycles = 0;

#if VM_TRACE
  std::unique_ptr<VerilatedVcdC> trace;
#endif
};

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_VERILATORSIMINTERFACE_H
