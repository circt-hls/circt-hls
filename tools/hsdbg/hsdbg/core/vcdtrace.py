from collections import defaultdict
from hsdbg.core.trace import *

# @TODO: VCDVCD is very slow for large files since it builds an in-memory
# representation of the VCD trace. Low hanging fruit could be to index the
# VCD trace and build a form of "finger table" for correlating step locations
# to file locations. This can then be used as a basis for a fast runtime
# forward search for the exact timestep.
from vcdvcd import VCDVCD


def joinVCDName(lhs, rhs):
  # Joins two hierarhical VCD names.
  return f"{lhs}.{rhs}"


def getLeafVCDName(name):
  return name.split(".")[-1]


def resolveVCDHierarchy(vcd):
  """ Generates an instance hierarchy from the source vcd object. This is needed due to the
    function missing an implementation in VCDVCD."""
  hier = {}
  for sig in vcd.signals:
    curHier = hier
    for module in sig.split(".")[:-1]:
      if module not in curHier:
        curHier[module] = {}
      curHier = curHier[module]
  return hier


class VCDInstance:
  """ This class represents an instance of a module in a VCD file.
  """

  def __init__(self, name, signals, subinstances):
    self.signals = signals
    self.subinstances = subinstances
    self.name


class VCDTrace(Trace):
  """ A trace interface for VCD files.
  """

  def __init__(self, filename):
    super().__init__(filename)

  def query(self, signal, step):
    # A signal value is read from the VCD trace based on its hierarchical
    # name and the step value.
    return self.vcd[self.signalMap[signal.getHierName()]][step]

  def index(self):
    # We use VCDVCD as the VCD parsing library.
    self.vcd = VCDVCD(self.filename)

    # Infer top module name
    hier = resolveVCDHierarchy(self.vcd)
    if len(hier) != 1:
      raise Exception("Expected exactly one top-level in VCD hierarchy!")
    top = list(hier.keys())[0]

    def indexModule(instanceName, parentInstance):
      # hierarchical name of the instance in the VCD file.
      vcdHierInstanceName = instanceName if not parentInstance else joinVCDName(
          parentInstance.getHierName(), instanceName)
      instance = Instance(name=getLeafVCDName(vcdHierInstanceName),
                          signals=[],
                          parent=parentInstance)

      # Find all items under this instance
      items = [s for s in self.vcd.signals if s.startswith(vcdHierInstanceName)]

      # Filter those which are in submodules. For this, we just check if
      # there any any more '.' characters in the name.
      vcdSignals = [
          s for s in items
          if s.replace(f"{vcdHierInstanceName}.", "").count(".") == 0
      ]

      # Create Signal objects from the VCD signals in this instance.
      signals = []
      for vcdSignal in vcdSignals:
        sigObj = Signal(name=getLeafVCDName(vcdSignal), parent=instance)
        # Register the signal in the signal map based on the signal hierarchical
        # name.
        self.signalMap[sigObj.getHierName()] = vcdSignal
        signals.append(sigObj)

      # Associate the signals with the newly created Instance object.
      instance.signals = signals
      instance.parent = parentInstance

      return instance

    def recurseModuleHierarchy(instanceName,
                               instanceSubHier,
                               parentInstance=None):
      thisInstance = indexModule(instanceName, parentInstance)
      if instanceSubHier != {}:
        for subInstanceName in instanceSubHier:
          thisInstance.children.append(
              recurseModuleHierarchy(subInstanceName,
                                     instanceSubHier[subInstanceName],
                                     parentInstance=thisInstance))
      return thisInstance

    # Recursively index the VCD file, starting from the top-level.
    return recurseModuleHierarchy(top, hier[top])

  def getStartTime(self):
    return self.vcd.begintime

  def getEndTime(self):
    # Returns the last timestep of the simulation
    return self.vcd.endtime
