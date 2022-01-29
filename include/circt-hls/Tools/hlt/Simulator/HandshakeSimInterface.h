#ifndef CIRCT_TOOLS_HLT_HANDSHAKESIMINTERFACE_H
#define CIRCT_TOOLS_HLT_HANDSHAKESIMINTERFACE_H

#include "circt-hls/Tools/hlt/Simulator/VerilatorSimInterface.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace circt {
namespace hlt {

struct TransactableTrait {
  enum State {
    // No transaction
    Idle,
    // Transaction is in progress (expecting Transacted to be set during eval())
    TransactNext,
    // Just transacted.
    Transacted
  };
  // If this is set, the port was transacted in the last cycle.
  State txState = Idle;
  bool transacted() { return txState == State::Transacted; }
};

template <typename TSimPort>
struct HandshakePort : public TSimPort, public TransactableTrait {
  HandshakePort(CData *readySig, CData *validSig)
      : readySig(std::make_unique<VerilatorSignal<CData>>(readySig)),
        validSig(std::make_unique<VerilatorSignal<CData>>(validSig)){};
  HandshakePort(const std::string &name, CData *readySig, CData *validSig)
      : name(name),
        readySig(std::make_unique<VerilatorSignal<CData>>(readySig)),
        validSig(std::make_unique<VerilatorSignal<CData>>(validSig)){};

  void dump(std::ostream &out) const {
    out << (name.empty() ? "?" : name) << ">\t";
    out << "r: " << static_cast<int>(*readySig)
        << "\tv: " << static_cast<int>(*validSig);
  }
  bool valid() { return *this->validSig == 1; }
  bool ready() { return *this->readySig == 1; }

  std::unique_ptr<VerilatorSignal<CData>> readySig;
  std::unique_ptr<VerilatorSignal<CData>> validSig;
  std::string name;
};

struct HandshakeInPort : public HandshakePort<SimulatorInPort> {
  using HandshakePort<SimulatorInPort>::HandshakePort;
  void reset() override { *(this->validSig) = !1; }
  bool ready() override {
    // An input port is ready to accept inputs when an input is not already
    // pushed onto the port (validSig == 1).
    return *(this->validSig) == 0;
  }

  // Writing to an input port implies setting the valid signal.
  virtual void write() { *(this->validSig) = !0; }

  /// An input port transaction is fulfilled by de-asserting the valid (output)
  // signal of his handshake bundle.
  bool eval(bool firstInStep) override {
    bool changed = false;
    if (txState == Transacted)
      txState = Idle;

    bool handshaking = *(this->validSig) && *(this->readySig);
    bool transacted = txState == TransactNext && firstInStep;
    if (handshaking || transacted) {
      if (transacted) {
        changed |= this->validSig->assign(0);
        txState = Transacted;
        // Transactions are considered a valid keepAlive reason.
        this->keepAlive();
      } else
        txState = TransactNext;
    } else
      txState = Idle;
    return changed;
  }
};

struct HandshakeOutPort : public HandshakePort<SimulatorOutPort> {
  using HandshakePort<SimulatorOutPort>::HandshakePort;
  void reset() override { *(this->readySig) = !1; }
  virtual void read() {
    // todo
  }

  /// An output port transaction is fulfilled whenever the ready signal of the
  // handshake bundle is asserted and the valid signal is not. A precondition
  // is that the valid signal was asserter before the ready signal.
  bool eval(bool firstInStep) override {
    if (txState == Transacted)
      txState = Idle;

    bool handshaking = *(this->validSig) && *(this->readySig);
    bool transacted = txState == TransactNext && firstInStep;
    if (handshaking || transacted) {
      if (transacted) {
        // *this->readySig = 0;
        txState = Transacted;
        // Transactions are considered a valid keepAlive reason.
        this->keepAlive();
      } else
        txState = TransactNext;
    } else
      txState = Idle;

    // Implementing port determines whether there are any actual state changes
    // on the port signals.
    return false;
  }
};

template <typename TData, typename THandshakeIOPort>
struct HandshakeDataPort : public THandshakeIOPort {
  HandshakeDataPort() {}
  HandshakeDataPort(CData *readySig, CData *validSig, TData *dataSig)
      : THandshakeIOPort(readySig, validSig), dataSig(dataSig){};
  HandshakeDataPort(const std::string &name, CData *readySig, CData *validSig,
                    TData *dataSig)
      : THandshakeIOPort(name, readySig, validSig), dataSig(dataSig){};
  void dump(std::ostream &out) const {
    THandshakeIOPort::dump(out);
    out << "\t" << static_cast<int>(*dataSig);
  }
  TData *dataSig = nullptr;
};

template <typename TData>
struct HandshakeDataInPort : HandshakeDataPort<TData, HandshakeInPort> {
  using HandshakeDataPortImpl = HandshakeDataPort<TData, HandshakeInPort>;
  using HandshakeDataPortImpl::HandshakeDataPortImpl;
  void writeData(TData in) {
    HandshakeDataPortImpl::write();
    *(this->dataSig) = in;
  }
};

template <typename TData>
struct HandshakeDataOutPort : HandshakeDataPort<TData, HandshakeOutPort> {
  using HandshakeDataPortImpl = HandshakeDataPort<TData, HandshakeOutPort>;
  using HandshakeDataPortImpl::HandshakeDataPortImpl;
  TData readData() {
    HandshakeDataPortImpl::read();
    return *(this->dataSig);
  }
};

// A HandshakeMemoryInterface represents a wrapper around a
// handshake.extmemory operation. It is initialized with a set of load- and
// store ports which, when transacting, will access the pointer provided to
// the memory interface during simulation. The memory interface inherits from
// SimulatorInPort due to handshake circuits receiving a memory interface as a
// memref input.

// When iterating over ports in writeInputRec, we dynamic_cast them to check for
// their type. This is difficult to do if we dyn_cast on
// HandshakeMemoryInterface due to TAddr (not directly known from a value that
// is pushed onto the simulator).
template <typename TData>
class HandshakeMemoryInterfaceBase : public SimulatorInPort,
                                     public TransactableTrait {
protected:
  // The memory pointer is set by the simulation engine during execution.
  TData *memory_ptr = nullptr;

public:
  void setMemory(void *memory) {
    if (memory_ptr != nullptr)
      assert(memory_ptr == memory &&
             "The memory should always point to the same base address "
             "throughout simulation.");
    memory_ptr = reinterpret_cast<TData *>(memory);
    txState = TransactNext;
  }
};

template <typename TData, typename TAddr>
class HandshakeMemoryInterface : public HandshakeMemoryInterfaceBase<TData> {

  // The size of the memory associated with this interface.
  size_t memorySize;

  void write(const TAddr &addr, const TData &data) {
    assert(this->memory_ptr != nullptr && "Memory not set.");
    assert(addr < memorySize && "Address out of bounds.");
    this->memory_ptr[addr] = data;
  }

  TData read(const TAddr &addr) {
    assert(this->memory_ptr != nullptr && "Memory not set.");
    assert(addr < memorySize && "Address out of bounds.");
    return this->memory_ptr[addr];
  }

  class MemoryPortBundle : public SimulatorPort {
  public:
    MemoryPortBundle(const std::vector<SimulatorPort *> &ports) : ports(ports) {
      clearTransacted();
    }
    std::vector<SimulatorPort *> ports;

    /// Evaluates each of the ports in this bundle. This interacts with the
    /// propagate(...) function in that the transacted flag will be set to true
    /// after transaction occured.
    bool eval(bool firstInStep) override {
      bool changed = false;
      for (auto &p : ports) {
        changed |= p->eval(firstInStep);
        auto transactable = dynamic_cast<TransactableTrait *>(p);
        assert(transactable);
        if (transactable->transacted()) {
          portTransacted(p);
          p->keepAlive();
        }
      }
      return changed;
    }

    bool hasTransacted(SimulatorPort *p) const { return transacted.at(p); }

    void reset() override {}

    // Propagation function for this memory bundle. This is where we'll
    // implement the combinational logic for the port, pre-rising edge.
    // Returns true if any signals changed.
    virtual bool propagate(HandshakeMemoryInterface &mem) = 0;

    void setKeepAliveCallback(const KeepAliveFunction &f) override {
      for (auto &p : ports)
        p->setKeepAliveCallback(f);
      SimulatorPort::setKeepAliveCallback(f);
    }

    // Ready whenever none of the ports have transacted.
    bool ready() {
      return llvm::all_of(transacted, [](auto &p) { return !p.second; });
    }

  private:
    // Register that 'port' transacted. Upon all ports being transacted, this
    // will clear the 'transacted' map. This is how we maintain the fork-like
    // logic of the memory interface.
    void portTransacted(SimulatorPort *port) {
      transacted.at(port) = true;
      if (llvm::all_of(transacted, [](const auto &p) { return p.second; }))
        clearTransacted();
    }
    void clearTransacted() {
      transacted.clear();
      for (auto &p : ports)
        transacted[p] = false;
    }

    // Maintain a map of each port and whether it has been transacted in the
    // current transaction.
    std::map<SimulatorPort *, bool> transacted = {};
  };

  struct StorePort : MemoryPortBundle {
    StorePort(const std::shared_ptr<HandshakeDataOutPort<TData>> &data,
              const std::shared_ptr<HandshakeDataOutPort<TAddr>> &addr,
              const std::shared_ptr<HandshakeInPort> &done)
        : MemoryPortBundle({data.get(), addr.get(), done.get()}), data(data),
          addr(addr), done(done){};
    std::shared_ptr<HandshakeDataOutPort<TData>> data;
    std::shared_ptr<HandshakeDataOutPort<TAddr>> addr;
    std::shared_ptr<HandshakeInPort> done;
    bool storeNext = false;
    TData nextData = 0;
    TAddr nextAddr = 0;

    // Store port transaction rules:
    // Whenever data and address are valid, stores data into the memory.
    // This causes done.valid to be asserted. After we've transacted done,
    // we can then transact the data and address ports.
    bool propagate(HandshakeMemoryInterface &mem) override {
      bool changed = false;

      // Ready mode implies address and data signals are ready.
      if (this->ready()) {
        changed |= addr->readySig->assign(1);
        changed |= data->readySig->assign(1);
      }

      // Deassert ready signals on address and data once transacted
      if (this->hasTransacted(addr.get()))
        changed |= addr->readySig->assign(0);
      if (this->hasTransacted(data.get()))
        changed |= data->readySig->assign(0);

      // Store the fact that we will write into the memory on the _next_ clock
      // cycle. We do not do this immediately; a simple example of when this
      // goes wrong is:
      //  for (int i = 0; i < N; i++) {
      //    if (a[i] == n)
      //      a[i] = 0;
      //  }
      // If we store a[i] = 0 immediately, then a[i], when asynchronously loaded
      // from memory, will be 0, and the if statement is then no longer true!
      if (*(addr->validSig) && (*data->validSig)) {
        storeNext = true;
        nextData = *(data->dataSig);
        nextAddr = *(addr->dataSig);
      } else
        storeNext = false;

      // After address and data has transacted, the done signal is valid,
      // mimicking a 1 cycle delay through the memory.
      if (this->hasTransacted(addr.get()) && this->hasTransacted(data.get()))
        changed |= done->validSig->assign(1);

      return changed;
    }

    bool eval(bool firstInStep, HandshakeMemoryInterface &mem) {
      // If we have a valid address and data in the previous cycle, store the
      // data into the memory.
      if (firstInStep && storeNext) {
        mem.write(nextAddr, nextData);
        storeNext = false;
      }
      return MemoryPortBundle::eval(firstInStep);
    }
  };

  struct LoadPort : MemoryPortBundle {

    LoadPort(const std::shared_ptr<HandshakeDataInPort<TData>> &data,
             const std::shared_ptr<HandshakeDataOutPort<TAddr>> &addr,
             const std::shared_ptr<HandshakeInPort> &done)
        : MemoryPortBundle({data.get(), addr.get(), done.get()}), data(data),
          addr(addr), done(done){};
    std::shared_ptr<HandshakeDataInPort<TData>> data;
    std::shared_ptr<HandshakeDataOutPort<TAddr>> addr;
    std::shared_ptr<HandshakeInPort> done;

    // Load port transaction rules:
    // Whenever the addr.valid, raise addr.ready (transact), and raise
    // data.valid and done.valid. Keep these high until both of the ports
    // transacted. These two ports may transact at different times, so state
    // must be maintained.
    bool propagate(HandshakeMemoryInterface &mem) override {
      bool changed = false;

      if (this->ready()) {
        changed |= addr->readySig->assign(0);
      }

      if (*(addr->validSig)) {
        // It should always be legal to read the memory when the address signal
        // is valid.
        size_t addrValue = *(addr->dataSig);
        lastReadData = mem.read(addrValue);
      }

      if (*(addr->validSig)) {
        if (!this->hasTransacted(data.get())) {
          changed |= data->validSig->assign(1);
          *(data->dataSig) = lastReadData;
        } else
          changed |= data->validSig->assign(0);

        if (!this->hasTransacted(done.get()))
          changed |= done->validSig->assign(1);
        else
          changed |= done->validSig->assign(0);
      }

      // Set address ready whenever we'll have transacted both the data and
      // done ports in the next cycle.
      if (this->hasTransacted(done.get()) && this->hasTransacted(data.get()))
        changed |= addr->readySig->assign(1);

      return changed;
    }

    // Store the last read data value when the address was valid. It may occur
    // that the address signal handshakes before the data signal, and then
    // changes in value, so we can only the the value that was read by a valid
    // address signal.
    TData lastReadData;
  };

public:
  // A memory interface is initialized with a static memory size. This is
  // generated during wrapper generation.
  HandshakeMemoryInterface(size_t size) : memorySize(size) {}

  // Forward keepAlive callback to memory ports
  void setKeepAliveCallback(const KeepAliveFunction &f) override {
    this->keepAlive = f;
    for (auto &port : storePorts)
      port.setKeepAliveCallback(f);
    for (auto &port : loadPorts)
      port.setKeepAliveCallback(f);
  }

  void dump(std::ostream &os) const {}

  virtual ~HandshakeMemoryInterface() = default;

  void
  addStorePort(const std::shared_ptr<HandshakeDataOutPort<TData>> &dataPort,
               const std::shared_ptr<HandshakeDataOutPort<TAddr>> &addrPort,
               const std::shared_ptr<HandshakeInPort> &donePort) {
    storePorts.push_back(StorePort(dataPort, addrPort, donePort));
  }

  void addLoadPort(const std::shared_ptr<HandshakeDataInPort<TData>> &dataPort,
                   const std::shared_ptr<HandshakeDataOutPort<TAddr>> &addrPort,
                   const std::shared_ptr<HandshakeInPort> &donePort) {
    loadPorts.push_back(LoadPort(dataPort, addrPort, donePort));
  }

  void reset() override {
    for (auto &port : storePorts) {
      *(port.data->validSig) = !1;
      *(port.addr->validSig) = !1;
      *(port.done->readySig) = !1;
    }
    for (auto &port : loadPorts) {
      *(port.data->readySig) = !1;
      *(port.addr->validSig) = !1;
      *(port.done->readySig) = !1;
    }
  }
  bool ready() override {
    assert(false && "N/A for memory interfaces.");
    return false;
  }

  // Writing to an input port implies setting the valid signal.
  virtual void write() { assert(false && "N/A for memory interfaces."); }

  /// An input port transaction is fulfilled by de-asserting the valid
  /// (output)
  // signal of his handshake bundle.
  bool eval(bool firstInStep) override {
    bool changed = false;
    switch (this->txState) {
    case TransactableTrait::Idle:
      break;
    case TransactableTrait::TransactNext:
      this->txState = TransactableTrait::Transacted;
      break;
    case TransactableTrait::Transacted:
      this->txState = TransactableTrait::Idle;
      break;
    }

    // Current cycle transactions:
    // Load ports
    for (auto &loadPort : loadPorts)
      changed |= loadPort.propagate(*this);

    // Store ports
    for (auto &storePort : storePorts)
      changed |= storePort.propagate(*this);

    // Evaluate the ports
    for (auto &port : storePorts)
      changed |= port.eval(firstInStep, *this);
    for (auto &port : loadPorts)
      changed |= port.eval(firstInStep);

    return changed;
  }

private:
  std::vector<StorePort> storePorts;
  std::vector<LoadPort> loadPorts;
};

template <typename TInput, typename TOutput, typename TModel>
class HandshakeSimInterface
    : public VerilatorSimInterface<TInput, TOutput, TModel> {
public:
  using VerilatorSimImpl = VerilatorSimInterface<TInput, TOutput, TModel>;

  template <typename T>
  struct TransactionBuffer {
    TransactionBuffer(const T &data = T()) : data(data) {
      for (int i = 0; i < std::tuple_size<T>(); ++i)
        transacted[i] = false;
    }
    T data;
    // Maintain a mapping between the index of each subtype in data and
    // whether that subtype has been transacted.
    std::map<unsigned, bool> transacted;
    // Flag to indicate if the input control has been transacted for this
    // buffer.
    bool transactedControl = false;
  };

  struct InputBuffer : public TransactionBuffer<TInput> {
    InputBuffer(const TInput &data) : TransactionBuffer<TInput>(data) {}

    bool done() {
      return this->transactedControl &&
             std::all_of(this->transacted.begin(), this->transacted.end(),
                         [](const auto &pair) { return pair.second; });
    }
  };

  struct OutputBuffer : public TransactionBuffer<TOutput> {
    OutputBuffer() : TransactionBuffer<TOutput>() {}

    bool valid() {
      return this->transactedControl &&
             std::all_of(this->transacted.begin(), this->transacted.end(),
                         [](const auto &pair) { return pair.second; });
    }
  };

  HandshakeSimInterface() : VerilatorSimImpl() {}

  // The handshake simulator is ready to accept inputs whenever it is not
  // currently transacting an input buffer.
  bool inReady() override { return !this->inBuffer.has_value(); }

  // The handshake simulator is ready to provide an output whenever it has
  // a valid output buffer.
  bool outValid() override { return this->outBuffer.valid(); }

  void evaluate(bool risingEdge) {
    // Evaluate the ports until no more changes occur.
    bool changed = true;
    int changeCount = HLT_TIMEOUT;
    int finalEval = 2;
    while (changed || finalEval) {
      if (changeCount-- == 0) {
        std::cerr << "Evaluated handshake sim interface HLT_TIMEOUT timeout "
                     "times; this probably means that there is a combinational "
                     "loop between the simulator interface logic and the RTL "
                     "simulation.\n";
        assert(false);
      }
      changed = false;
      this->advanceTime();
      // Transact all I/O ports
      for (auto &port : llvm::enumerate(this->outPorts)) {
        changed |= port.value()->eval(risingEdge);
        auto transactable =
            dynamic_cast<TransactableTrait *>(port.value().get());
        assert(transactable);
        if (transactable->transacted())
          outBuffer.transacted[port.index()] = true;
      }
      for (auto &port : llvm::enumerate(this->inPorts)) {
        changed |= port.value()->eval(risingEdge);
        auto transactable =
            dynamic_cast<TransactableTrait *>(port.value().get());
        assert(transactable);
        if (transactable->transacted())
          inBuffer.value().transacted[port.index()] = true;
      }

      // Transact control ports
      changed |= inCtrl->eval(risingEdge);
      if (inCtrl->transacted())
        inBuffer.value().transactedControl = true;

      changed |= outCtrl->eval(risingEdge);
      if (outCtrl->transacted())
        outBuffer.transactedControl = true;

      // This can no longer be the rising edge
      risingEdge = false;

      // If no change occured then we run the evaluation one final time, to let
      // the model react to any sim changes.
      if (!changed)
        finalEval--;
      else
        finalEval = 2;
    }
  }

  void step() override {
    // Rising edge
    VerilatorSimImpl::clock_rising();

    readToOutputBuffer();
    writeFromInputBuffer();
    evaluate(/*risingEdge=*/true);

    // Set output ports readyness based on which outputs in the current output
    // buffer have been transacted. This also means that we'll start off with
    // all outputs ready, and gradually reduce the number of outputs that are
    // ready until all ports are transacted. When the output buffer is then
    // ready, all of these ready signals will go high again.
    for (auto outPort : llvm::enumerate(this->outPorts)) {
      auto outPortp = dynamic_cast<HandshakeOutPort *>(outPort.value().get());
      assert(outPortp);
      *(outPortp->readySig) = !outBuffer.transacted[outPort.index()];
    }
    *(outCtrl->readySig) = !outBuffer.transactedControl;

    // Falling edge
    VerilatorSimImpl::clock_falling();
    evaluate(/*risingEdge=*/false);
    this->advanceTime();
    this->m_clockCycles++;
  }

  void setup() override {
    inCtrl->name = "inCtrl";
    outCtrl->name = "outCtrl";
    assert(inCtrl->readySig != nullptr && "Missing in control ready signal");
    assert(inCtrl->validSig != nullptr && "Missing in control valid signal");
    assert(outCtrl->readySig != nullptr && "Missing out control ready signal");
    assert(outCtrl->validSig != nullptr && "Missing out control valid signal");

    // Forward keepAlive callback to ports.
    for (auto &port : this->outPorts)
      port->setKeepAliveCallback(this->keepAlive);
    for (auto &port : this->inPorts)
      port->setKeepAliveCallback(this->keepAlive);
    inCtrl->setKeepAliveCallback(this->keepAlive);
    outCtrl->setKeepAliveCallback(this->keepAlive);

    // Do verilator initialization; this will reset the circuit
    VerilatorSimImpl::setup();

    // Run a few cycles to ensure everything works after the model is out of
    // reset and a subset of all ports are ready/valid.
    for (int i = 0; i < 2; ++i)
      VerilatorSimImpl::clock();
  }

  void dump(std::ostream &out) const override {
    out << "Control states:\n";
    out << *inCtrl << "\n";
    out << *outCtrl << "\n";
    VerilatorSimImpl::dump(out);
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
    if (!inBufferV.transacted[I]) {
      // Normal port?
      if (auto inPort = dynamic_cast<HandshakeDataInPort<decltype(value)> *>(p);
          inPort) {
        // A value can be written to an input port when it is not already
        // trying to transact a value.
        if (!inPort->valid()) {
          inPort->writeData(value);
        }
      }
      // Memory interface?
      else if (auto inMemPort = dynamic_cast<HandshakeMemoryInterfaceBase<
                   std::remove_pointer_t<decltype(value)>> *>(p);
               inMemPort) {
        inMemPort->setMemory(reinterpret_cast<void *>(value));
      } else {
        assert(false && "Unsupported input port type");
      }
    }

    writeInputRec<I + 1, Tp...>(tInput);
  }

  void writeFromInputBuffer() {
    if (!inBuffer.has_value())
      return; // Nothing to transact.

    auto &inBufferV = inBuffer.value();

    // Try writing input data.
    writeInputRec(inBufferV.data);

    // Try writing input control.
    if (!inBufferV.transactedControl)
      inCtrl->write();

    // Finish writing input buffer?
    if (inBufferV.done())
      inBuffer = std::nullopt;
  }

  void pushInput(const TInput &v) override {
    assert(!inBuffer.has_value() &&
           "pushing input while already having an input buffer?");
    inBuffer = {v};
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
    auto outPort = dynamic_cast<HandshakeDataOutPort<ValueType> *>(
        this->outPorts.at(I).get());
    assert(outPort);
    if (outPort->valid() && outPort->ready()) {
      std::get<I>(tOutput) = outPort->readData();
    }
    readOutputRec<I + 1, Tp...>(tOutput);
  }

  void readToOutputBuffer() {
    if (outBuffer.valid())
      return; // Nothing to transact.

    // Try reading output data.
    readOutputRec(outBuffer.data);

    // OutBuffer will be cleared by popOutput if all data has been read.
  }

  TOutput popOutput() override {
    assert(outBuffer.valid() && "popping output buffer that is not valid?");
    auto vOutput = outBuffer.data;
    outBuffer = OutputBuffer(); // reset
    return vOutput;
  }

protected:
  // Handshake interface signals. Defined as raw pointers since they are owned
  // by VerilatorSimInterface.
  std::unique_ptr<HandshakeInPort> inCtrl;
  std::unique_ptr<HandshakeOutPort> outCtrl;

  // In- and output buffers.
  // @todo: this could be made into separate buffers for each subtype within
  // TInput and TOutput, allowing for decoupling of starting the writing of a
  // new input buffer until all values within an input have been transacted.
  std::optional<InputBuffer> inBuffer;
  OutputBuffer outBuffer;
};

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_HANDSHAKESIMINTERFACE_H
