#ifndef CIRCT_TOOLS_HLT_CALYXSIMINTERFACE_H
#define CIRCT_TOOLS_HLT_CALYXSIMINTERFACE_H

#include "circt-hls/Tools/hlt/Simulator/MemoryInterface.h"
#include "circt-hls/Tools/hlt/Simulator/VerilatorSimInterface.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace circt {
namespace hlt {

template <typename TSig, typename TPort>
class CalyxPort : public TPort, public VerilatorSignal<TSig> {
public:
  using VerilatorSignal<TSig>::VerilatorSignal;
  bool eval(bool firstInStep) override {}
  void reset() override {}
};

template <typename TSig>
using CalyxOutPort = CalyxPort<TSig, SimulatorOutPort>;

template <typename TSig>
using CalyxInPort = CalyxPort<TSig, SimulatorInPort>;

template <typename TData, typename TAddr>
class CalyxMemoryInterface : public SimulatorInPort,
                             public MemoryInterfaceBase<TData> {

public:
  // A memory interface is initialized with a static memory size. This is
  // generated during wrapper generation.
  CalyxMemoryInterface(size_t size,
                       std::shared_ptr<CalyxOutPort<TData>> readDataSignal,
                       std::shared_ptr<CalyxOutPort<CData>> doneSignal,
                       std::shared_ptr<CalyxInPort<TData>> writeDataSignal,
                       std::shared_ptr<CalyxInPort<CData>> writeEnSignal,
                       std::shared_ptr<CalyxInPort<TAddr>> addrSignal)
      : MemoryInterfaceBase<TData>(size), readDataSignal(readDataSignal),
        addrSignal(addrSignal), doneSignal(doneSignal),
        writeDataSignal(writeDataSignal), writeEnSignal(writeEnSignal) {}
  void dump(std::ostream &os) const {}

  void reset() {
    *readDataSignal = 0;
    *writeDataSignal = 0;
    *doneSignal = 0;
    *writeEnSignal = 0;
    *addrSignal = 0;
  }

  // Writing to an input port implies setting the valid signal.
  virtual void write() { assert(false && "N/A for memory interfaces."); }
  bool eval(bool firstInStep) override {
    if (this->writeEnSignal)
      this->write(*addrSignal, *writeDataSignal);
    *readDataSignal = this->read(*addrSignal);
  }

private:
  std::shared_ptr<CalyxOutPort<TData>> readDataSignal;
  std::shared_ptr<CalyxOutPort<TData>> writeDataSignal;
  std::shared_ptr<CalyxInPort<TAddr>> addrSignal;
  std::shared_ptr<CalyxInPort<CData>> doneSignal;
  std::shared_ptr<CalyxInPort<CData>> writeEnSignal;
};

template <typename TInput, typename TOutput, typename TModel>
class CalyxSimInterface
    : public VerilatorSimInterface<TInput, TOutput, TModel> {
  using VerilatorSimImpl = VerilatorSimInterface<TInput, TOutput, TModel>;

public:
  CalyxSimInterface() : VerilatorSimImpl() {}

  // The Calyx simulator is ready to accept inputs whenever it is not
  // currently transacting an input buffer.
  bool inReady() override { return !this->inBuffer.has_value(); }

  // The Calyx simulator is ready to provide an output whenever it has
  // a valid output buffer.
  bool outValid() override { return this->outBuffer.has_value(); }

  void pushInput(const TInput &input) override {
    assert(!this->inBuffer.has_value());
    this->inBuffer = input;
  }

  TOutput popOutput() override {
    assert(this->outBuffer.has_value());
    TOutput out = this->outBuffer.value();
    this->outBuffer.reset();
    return out;
  }

  void step() override {
    // Rising edge
    VerilatorSimImpl::clock_rising();

    readToOutputBuffer();
    bool wroteInput = writeFromInputBuffer();
    VerilatorSimImpl::clock_falling();

    if (wroteInput) {
      // Reset the 'go' signal.
      this->go->assign(0);
    }

    this->advanceTime();
    this->m_clockCycles++;
  }

  template <std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  readOutputRec(std::tuple<Tp...> &) {
    // End-case, do nothing
  }

  template <std::size_t I = 0, typename... Tp>
      inline typename std::enable_if <
      I<sizeof...(Tp), void>::type readOutputRec(std::tuple<Tp...> &tOutput) {
    using ValueType = std::remove_reference_t<decltype(std::get<I>(tOutput))>;
    auto outPort =
        dynamic_cast<CalyxOutPort<ValueType> *>(this->outPorts.at(I).get());
    assert(outPort);
    std::get<I>(tOutput) = *outPort;
    readOutputRec<I + 1, Tp...>(tOutput);
  }

  void readToOutputBuffer() {
    if (*this->done == 0)
      return;
    // Kernel indicated 'done'; read to output buffer.
    assert(!this->outBuffer.has_value() && "Output buffer already has value");
    outBuffer = TOutput();
    readOutputRec(outBuffer.value());
  }

  template <std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  writeInputRec(const std::tuple<Tp...> &) {
    // End-case, do nothing
  }

  template <std::size_t I = 0, typename... Tp>
      inline
      typename std::enable_if < I<sizeof...(Tp), void>::type
                                writeInputRec(const std::tuple<Tp...> &tInput) {
    auto value = std::get<I>(tInput);
    auto &inBufferV = inBuffer.value();
    // Is this a simple input port?
    auto p = this->inPorts.at(I).get();
    // Normal port?
    if (auto inPort = dynamic_cast<CalyxInPort<decltype(value)> *>(p); inPort)
      inPort->assign(value);
    // Memory interface?
    else if (auto inMemPort = dynamic_cast<
                 MemoryInterfaceBase<std::remove_pointer_t<decltype(value)>> *>(
                 p);
             inMemPort) {
      inMemPort->setMemory(reinterpret_cast<void *>(value));
    } else {
      assert(false && "Unsupported input port type");
    }

    writeInputRec<I + 1, Tp...>(tInput);
  }

  // Writes a value from the input buffer to the ports of the model. Returns
  // true if the kernel was initiated.
  bool writeFromInputBuffer() {
    if (!this->inBuffer.has_value())
      return false;
    assert(!this->inBuffer.has_value() && "Input buffer already has a value!");
    auto &inBufferV = inBuffer.value();
    writeInputRec(inBufferV);

    // Finally, write the 'go' port.
    return this->go.get()->assign(1);
  }

protected:
  // Pointer to the "go" and "done" ports of the calyx component.
  std::shared_ptr<CalyxInPort<CData>> go, done;
  std::optional<TInput> inBuffer;
  std::optional<TOutput> outBuffer;
  bool running = false;
};

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_CALYXSIMINTERFACE_H