from dataclasses import dataclass
from collections import defaultdict
from hsdbg.core.model import *
import tempfile
import os

# Possible custom attributes; these are the attributes which may appear
# in the comment section after an edge definition in the .dot file.

# Which output port of the source node this edge is connected to.
CA_OUTPUT = "output"
# Which input port of the sink node this edge is connected to.
CA_INPUT = "input"


@dataclass(frozen=True, order=True)
class DotEdge:
  src: str  # name of source node
  dst: str  # name of destination node
  attributes: set  # keep as a list to make hashable
  line: int  # line number in the original file
  # custom attributes are attributes that are not part of the standard dot language.
  customAttributes: set

  def getCustomAttr(self, name):
    for attr in self.customAttributes:
      if attr[0] == name:
        return attr[1]
    return None

  def getAttr(self, name):
    for attr in self.attributes:
      if attr[0] == name:
        return attr[1]
    return None

  def __str__(self):
    """String representation of this edge in the DOT graph. This is used for
           generating the modified .dot file."""
    out = ""
    out += (f'\"{self.src}\" -> \"{self.dst}\"')
    if len(self.attributes) > 0:
      out += " ["
      for attr in self.attributes:
        out += f'{attr[0]}=\"{attr[1]}\" '
      out += "]"

    if len(self.customAttributes) > 0:
      out += "// "
      for attr in self.customAttributes:
        out += f'{attr[0]}=\"{attr[1]}\" '
    out += '\n'
    return out

  def isFromInput(self):
    # An input node will not have an output port.
    return self.getAttr(CA_OUTPUT) is None

  def isToOutput(self):
    # An output node will not have an input port.
    return self.getAttr(CA_INPUT) is None


class DotFile:
  """ A class which handles in-place modification of a dot file.
    This is unfortunately not available in any existing graphviz library.
    This class doesn't parse the vcd grammer. Rather, it looks for edges in the
    input file and registers these edges. It then allows the user to add
    attributes to edges and rewrite the file.
    """

  def __init__(self, filename):
    self.rawDot = []  # List of lines
    self.modifiedDot = []
    self.edgeToLine = {}
    self.lineToEdge = {}
    self.edges = set()
    self.parseDot(filename)
    self.reset()

  def reset(self):
    # Resets the modifiedDot to the rawDot.
    self.modifiedDot = self.rawDot

  def addAttributesToEdge(self, edge, attrs):
    if edge not in self.edges:
      raise Exception(f"Edge not found: {edge}")
    # Edges are frozen to allow for hashing, so we need to make a copy.
    edgeAttrs = set(edge.attributes)
    for attr in attrs:
      edgeAttrs.add(attr)
    self.modifiedDot[edge.line] = str(
        DotEdge(edge.src, edge.dst, edgeAttrs, edge.line,
                edge.customAttributes))

  def parseCustomAttributes(self, attrString):
    """ Parses custom attributes on a handshake graphviz edge."""
    attrList = set()
    for attr in attrString.split(" "):
      if "=" in attr:
        k, v = attr.split("=")
        v = v.replace('"', "")
        attrList.add(((k, v)))
    return frozenset(attrList)

  def parseDotEdge(self, i, line):
    """Parses a graphviz file edge. An edge has the format:
                "from" -> "to" [optional attributes] // custom attributes
        """
    src, rest = line.split("->")[0].strip(), line.split("->")[1].strip()

    rest = rest.split("//")
    customAttributes = frozenset()
    if len(rest) > 1:
      customAttributes = self.parseCustomAttributes(rest[1].strip())
    rest = rest[0].strip()
    dstSplit = rest.split("[")
    dst = dstSplit[0]
    src = src.replace('"', '').strip()
    dst = dst.replace('"', '').strip()
    # split [.* from dst and also return the second part
    attributes = frozenset()
    if len(dstSplit) > 1:
      attrStr = dstSplit[1].replace(']', '')
      # attributes is a list of key value pairs with '=' in between
      # and separated by spaces. Parse it as a list of pairs
      attributes = [x.strip() for x in attrStr.split(" ")]
      attributes = [x.split("=") for x in attributes if len(x) > 0]
      attributes = frozenset({
          (x[0].replace("\"", ""), x[1].replace("\"", "")) for x in attributes
      })
    edge = DotEdge(src, dst, attributes, i, customAttributes)
    self.edgeToLine[edge] = i
    self.lineToEdge[i] = edge
    self.edges.add(edge)

  def parseDot(self, filename):
    with open(filename, 'r') as f:
      self.rawDot = f.readlines()

    # Create edge to line mapping
    for i, line in enumerate(self.rawDot):
      if "->" in line:
        self.parseDotEdge(i, line)

  def dump(self, path):
    """ write the modified dot file to path."""
    with open(path, "w") as f:
      f.write("".join(self.modifiedDot))
