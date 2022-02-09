from hsdbg.frontends.dotmodel import *
from hsdbg.core.vcdtrace import *
from hsdbg.frontends.dotfile import *
from hsdbg.core.utils import *


class HandshakeModelEdge(DotModelEdge):
  """A HandshakeModelEdge represents a connection between a valid, ready and
  optional data signal in a trace, and a DotEdge in a dotfile.
  """

  def __init__(self, parentHSModelNode, valid, ready, data=None):
    super().__init__()
    self.parent = parentHSModelNode
    self.valid = valid
    self.ready = ready
    self.data = data

  def dotBaseName(self):
    """ Returns the base name of the handshake bundle in the dot file. We assume
    that the dot edge name is the same as that used in the VCD trace."""
    dotName = self.parent.getHierName() + "." + self.valid.name.rsplit(
        "_valid")[0]
    return dotName

  def resolveToDot(self, dotEdges):
    """
    Filters the 
    """
    # First, locate all of the candidate edges. These are the edges which
    # originate from the source of this handshake bundle.
    basename = self.dotBaseName()

    def checkEdge(edge):
      # Check if there exists a direct name match; this is true for in- and output variables
      # which, due to the single-use requirement of handshake circuits should
      # only ever have 1 edge with the variable as a source or destination.
      if edge.src == basename or edge.dst == basename:
        return True

      # Check if there exists an edge where the edge basename + source port or basename + sinkport
      # matches the handshake bundle basename.
      srcPort = edge.getCustomAttr(CA_OUTPUT)
      if srcPort and joinHierName(edge.src, srcPort) == basename:
        return edge

      dstPort = edge.getCustomAttr(CA_INPUT)
      if dstPort and joinHierName(edge.dst, dstPort) == basename:
        return edge

      return None

    # A bit of course filtering to disregard things which we know cannot be
    # this edge (edges of other parent modules).
    candidates = [e for e in dotEdges if checkEdge(e)]

    if len(candidates) == 0:
      # Could not resolve a .dot edge to the handshake bundle.
      return
    elif len(candidates) == 1:
      self.edge = candidates[0]

  def updateDot(self, trace, step, dot):
    """ Creates a set of attributes that are to be added to the dot edge to
        indicate the state of the edge.
        """
    if self.edge == None:
      # No edge was resolved for this handshake bundle.
      return

    attrs = []

    # Handshake state
    isReady = trace.query(self.ready, step) == "1"
    isValid = trace.query(self.valid, step) == "1"
    attrs.append(("penwidth", "3"))

    if isReady and isValid:
      attrs.append(("color", "green"))
    elif isReady or isValid:
      attrs.append(("color", "orange"))
      attrs.append(("arrowhead", "dot" if isValid else "odot"))
    else:
      attrs.append(("color", "red"))

    # Data value
    if self.data != None:
      data = trace.query(self.data, step)
      if data != "x":
        attrs.append(("label", formatBitString(data)))

    # Add the attributes to the dot edge object.
    dot.addAttributesToEdge(self.edge, attrs)


class HandshakeModelNode(DotModelNode):

  def __init__(self, instance, signals):
    """ A handshake model node directly represents an instance in the VCD trace.

    Args:
        instance ([type]): Reference to the instance in the trace that this node
            represents.
    """
    super().__init__()
    self.instance = instance
    self.signals = signals
    # List of HandshakeModelEdge objects emanating from this node.
    self.edges = []
    # A list of HandshakeModelNode objects representing the children of this node.
    self.children = []

  def getHierName(self):
    return self.instance.getHierName()

  def resolve(self, dotEdges):
    """ Identifies handshake bundle signals based on the set of VCD signals provided
    to this model node, and creates HandshakeModelEdge's from these.

    A bundle is identified by the presence of identically named signals with a (_valid/_ready)
    and optional "_data" suffix available.

    Once we've identified handshake bundles in the VCD file, we can then create
    a HandshakeModelEdge. This object is intended to model the connection between
    one such bundle of signals and an edge in the .dot file.

    """

    def getDimensionless(sig):
      # strip a number surrounded by square brackets from the end of 'sig'.
      return re.sub(r"(\[\d*(:\d*)?\])", "", sig.name)

    def getBasename(sig):
      baseSig = getDimensionless(sig)
      # strip occurences of ready valid and data string at the end of sig
      baseSig = baseSig.rsplit("_ready")[0]
      baseSig = baseSig.rsplit("_valid")[0]
      baseSig = baseSig.rsplit("_data")[0]
      return baseSig

    bundles = defaultdict(lambda: [])
    for sig in self.signals:
      bundles[getBasename(sig)].append(sig)

    for basename, bundle in bundles.items():

      def resolveBundleSig(name):
        opt = [x for x in bundle if getDimensionless(x).endswith(f"_{name}")]
        if len(opt) == 1:
          return opt[0]
        return None

      dataSig = resolveBundleSig("data")
      validSig = resolveBundleSig("valid")
      readySig = resolveBundleSig("ready")

      if validSig and readySig:
        self.edges.append(
            HandshakeModelEdge(parentHSModelNode=self,
                               valid=validSig,
                               ready=readySig,
                               data=dataSig))

      # Resolve bundles to the edges in the dot file.
      for edge in self.edges:
        edge.resolveToDot(dotEdges)

  def updateDot(self, trace, step, dot):
    # Update the state of the edges
    for edge in self.edges:
      edge.updateDot(trace, step, dot)

    for child in self.children:
      child.updateDot(trace, step, dot)


class HandshakeModel(DotModel):

  @staticmethod
  def name():
    return "handshake"

  @staticmethod
  def addArguments(subparser):
    # Initialize handshake arguments
    subparser.add_argument("--vcd", help="The vcdfile to use.", type=str)

    # Initialize dot model arguments
    DotModel.addArguments(subparser)

  def __init__(self, args) -> None:
    super().__init__(dot=args.dot, port=args.port)

    if not args.vcd:
      raise ValueError("No vcd file specified.")

    self.trace = VCDTrace(args.vcd)
    self.resolve()

    # Go!
    self.startImageServer()

  def resolve(self):
    """ This function resolves the handshake .dot file with the handshake trace.
    Each Instance in the trace is associated with a HandshakeModelNode, which represents
    the .dot node of that instance.
    Then, we call node.resolve(...) to further resolve the handshake signals nested
    within that bundle.
    """
    assert self.trace != None

    # Create the model node hierarchy from the VCD file, starting from the top
    # level instance in the trace.
    def createModelNode(instance):
      node = HandshakeModelNode(signals=instance.signals, instance=instance)

      # Resolve the node - this is where the magic happens! associate handshake
      # signal bundles to edges in the .dot file.
      node.resolve(dotEdges=self.dotFile.edges)
      for childInstance in instance.children:
        node.children.append(createModelNode(childInstance))
      return node

    self.topNode = createModelNode(self.getTrace().getTopInstance())

  def updateModel(self):
    """ Updates the model based on the current trace step. """

    # Updating is performed recursively through the top model node.
    # Each node will instruct its associated edges to modify the state of
    # the DotFile object.
    self.topNode.updateDot(self.trace, self.step, self.dotFile)

    # Finally, dump the modified dot file through the DotModel. This will
    # also reset the dot object to a clean state, ready for the next step.
    self.dotUpdateFinished()
