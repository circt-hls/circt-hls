#ifndef CIRCT_TOOLS_HLT_MEMORYINTERFACE_H
#define CIRCT_TOOLS_HLT_MEMORYINTERFACE_H

#include <optional>

template <typename TData>
class MemoryInterfaceBase {
protected:
  // The memory pointer is set by the simulation engine during execution.
  TData *memory_ptr = nullptr;

public:
  MemoryInterfaceBase(std::optional<unsigned> memorySize = std::nullopt) {
    memorySize = memorySize;
  }

  virtual void setMemory(void *memory) {
    if (memory_ptr != nullptr)
      assert(memory_ptr == memory &&
             "The memory should always point to the same base address "
             "throughout simulation.");
    memory_ptr = reinterpret_cast<TData *>(memory);
  }

  void write(unsigned addr, const TData &data) {
    assert(this->memory_ptr != nullptr && "Memory not set.");
    if (memorySize.has_value())
      assert(addr < memorySize && "Address out of bounds.");
    this->memory_ptr[addr] = data;
  }

  TData read(unsigned addr) {
    assert(this->memory_ptr != nullptr && "Memory not set.");
    if (memorySize.has_value())
      assert(addr < memorySize && "Address out of bounds.");
    return this->memory_ptr[addr];
  }

protected:
  // A memory may be registered with a size. This is optional, and only applies
  // to statically sized memories.
  std::optional<unsigned> memorySize;
};

#endif // CIRCT_TOOLS_HLT_MEMORYINTERFACE_H