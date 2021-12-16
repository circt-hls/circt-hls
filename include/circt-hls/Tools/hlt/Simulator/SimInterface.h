#ifndef CIRCT_TOOLS_HLT_SIMINTERFACE_H
#define CIRCT_TOOLS_HLT_SIMINTERFACE_H

#include <cassert>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <list>
#include <mutex>

namespace circt {
namespace hlt {

#ifndef NDEBUG
#define debugOut std::cout
#else
#define debugOut 0 && std::cout
#endif

template <typename T, template <typename...> class Tmpl>
struct is_instance_of_template : std::false_type {};
template <template <typename...> class Tmpl, typename... Args>
struct is_instance_of_template<Tmpl<Args...>, Tmpl> : std::true_type {};

/// A simple atomic queue implementation.
template <typename T>
struct AtomicQueue {
  void push(const T &v) {
    std::lock_guard<std::mutex> l(lock);
    list.push_back(v);
  }

  T pop() {
    std::lock_guard<std::mutex> l(lock);
    assert(!list.empty() && "Trying to pop an empty queue");
    auto v = list.front();
    list.pop_front();
    return v;
  }

  unsigned size() {
    std::lock_guard<std::mutex> l(lock);
    return list.size();
  }

  bool empty() { return size() == 0; }

  /// Leaving this public for now...
  std::list<T> list;
  std::mutex lock;
};

template <typename TInput, typename TOutput>
struct SimQueues {
  AtomicQueue<TInput> in;
  AtomicQueue<TOutput> out;
  AtomicQueue<std::shared_ptr<std::condition_variable>> outReq;
};

using KeepAliveFunction = std::function<void()>;

/// Base class for simulator-related classes.
class SimBase {
public:
  /// Dump the state of the object.
  virtual void dump(std::ostream &os) const {}

  virtual void setKeepAliveCallback(const KeepAliveFunction &f) {
    assert(keepAlive == nullptr && "keepAlive callback already set!");
    keepAlive = f;
  }

  // A function that should be called by the simulator whenever a meaningful
  // state change has occured within the simulator. This is used to inform the
  // simulator driver that the simulator has progressed and subsequently
  // reset the timeout counter.
  KeepAliveFunction keepAlive;
};

std::ostream &operator<<(std::ostream &out, const SimBase &b) {
  b.dump(out);
  return out;
}

// A SimulatorPort represents a mapping between a software-like in/output
// variable and its underlying representation in the simulator.
struct SimulatorPort : SimBase {
  virtual ~SimulatorPort() = default;

  // Resets a port to its initial state.
  virtual void reset() = 0;

  // Evaluates a port. firstInStep is set whenever this is the first eval call
  // during the current timestep. For synchronous/clocked simulations, this can
  // be seen as the rising edge evaluation. Returns true if the port performed a
  // state change which may require the simulation to update in relation to it.
  virtual bool eval(bool firstInStep) = 0;
};

// A SimulatorInPort represents a mapping from a software-like input variable
// to its underlying representation in the simulator.
struct SimulatorInPort : public SimulatorPort {
  virtual bool ready() = 0;
};

// A SimulatorOutPort represents a mapping from a software-like output variable
// to its underlying representation in the simulator.
struct SimulatorOutPort : public SimulatorPort {
  virtual bool valid() = 0;
};

template <typename TInput, typename TOutput>
class SimInterface : public SimBase {
public:
  /// Steps the simulator; step semantics is simulator defines, as longs as it
  /// adheres with the semantics of "progressing the simulation".
  virtual void step() = 0;

  /// Returns true if the model is ready to accept an input.
  virtual bool inReady() = 0;

  /// Returns true if the model is ready to provide an output.
  virtual bool outValid() = 0;

  /// Push an input to the simulator.
  virtual void pushInput(const TInput &input) = 0;

  /// Pop an output from the simulator.
  virtual TOutput popOutput() = 0;

  /// The setup function will be called post-construction of the simulator.
  virtual void setup() = 0;

  /// The finish function will be called before deletion of the simulator.
  virtual void finish() = 0;

  /// Returns the current timestep of the simulator.
  virtual uint64_t time() = 0;

  template <typename T, typename... Args>
  T *addInputPort(Args... args) {
    static_assert(std::is_base_of<SimulatorInPort, T>::value,
                  "Port must inherit from SimulatorInPort");
    auto ptr = new T(args...);
    inPorts.push_back(std::move(std::unique_ptr<SimulatorInPort>(ptr)));
    return ptr;
  }

  template <typename T, typename... Args>
  T *addOutputPort(Args... args) {
    static_assert(std::is_base_of<SimulatorOutPort, T>::value,
                  "Port must inherit from SimulatorOutPort");
    auto ptr = new T(args...);
    outPorts.push_back(std::move(std::unique_ptr<SimulatorOutPort>(ptr)));
    return ptr;
  }

protected:
  // There should be exactly as many inPorts as the software interface of this
  // simulator has input arguments.
  std::vector<std::unique_ptr<SimulatorInPort>> inPorts;

  // There should be exactly as many outPorts as the software interface of this
  // simulator has output arguments.
  std::vector<std::unique_ptr<SimulatorOutPort>> outPorts;
};

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_SIMINTERFACE_H
