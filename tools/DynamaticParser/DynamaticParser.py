import networkx as nx
from networkx.drawing.nx_pydot import read_dot
import sys
import argparse
import os
import re
import tempfile

from networkx.readwrite.json_graph import tree


class ModuleWriter:

  def __init__(self, outstream):
    self.outstream = outstream
    self.indent = 0

  def indentize(self, string):
    return ' ' * self.indent + string

  def writeLine(self, string):
    self.outstream.write(self.indentize(string) + "\n")

  def writeIndent(self):
    self.outstream.write(' ' * self.indent)

  def writeValue(self, value):
    self.outstream.write(f"%{value}")

  def writeRegion(self, pre, continuation):
    self.writeLine(pre + " {")
    self.indent += 2
    continuation()
    self.indent -= 2
    self.writeLine("}")

  def write(self, string):
    self.outstream.write(string)


# A class to help resolve types. This is mainly used to add type hints so that
# we can resolve index types. This uses some very basic type inference rules on
# the handshake operations to propagate types.
class TypeResolver:

  def __init__(self):
    self.knownIndexValues = {}
    self.valueTypes = {}

  def setValueType(self, SSAName, width=0, index=False):
    if index:
      self.knownIndexValues[SSAName] = True
      self.valueTypes[SSAName] = "index"
    elif width == 0:
      self.valueTypes[SSAName] = "none"
    else:
      self.valueTypes[SSAName] = f"i{width}"

  def getType(self, SSAName):
    if SSAName not in self.valueTypes:
      raise Exception(f"Unknown value type for {SSAName}")
    return self.valueTypes[SSAName]

  def resolveTypesRecurse(self, ops):
    # Don't need to do anything here yet
    return

  def resolveTypes(self, ops):
    # initialize the type value mapping with integer and none types
    for op in ops.values():
      for resIdx, resSSA in op.results.items():
        self.setValueType(SSAName=resSSA, width=op.outs[resIdx])

    # Add statically known type conversion when going to MLIR
    for op in ops.values():
      if op.type == "CntrlMerge":
        self.setValueType(op.results[1], index=True)
      elif op.type == "Source":
        # Source ops just provide a token in MLIR
        self.setValueType(op.results[0], width=0)

    # Iterate type inference until a fixed point is reached
    self.resolveTypesRecurse(ops)


typeResolver = TypeResolver()


def parseIO(s):
  return


to_handshake_opmap = {
    "Constant": "constant",
    "Branch": "br",
    "Fork": "fork",
    "Sink": "sink",
    "Join": "join",
    "CntrlMerge": "control_merge",
    "Branch": "cond_br",
    "Mux": "mux",
    "Merge": "merge",
    "Buffer": "buffer",
    "Source": "source"
}


def to_handshake_op(type):
  if type not in to_handshake_opmap:
    raise RuntimeError("Unknown operation type: " + type)
  return to_handshake_opmap[type]


mlir_operator_map = {
    "add_op": "arith.addi",
    "icmp_ult_op": "arith.cmpi ult,",
    "mul_op": "arith.muli",
    "sub_op": "arith.subi",
    "shl_op": "arith.shli",
    "shr_op": "arith.shri",
    "and_op": "arith.andi",
    "or_op": "arith.ori",
    "xor_op": "arith.xori",
    "ret_op": "return"
}


def mlir_operator_type(type):
  if type not in mlir_operator_map:
    raise RuntimeError("Unknown operation type: " + type)
  return mlir_operator_map[type]


# Global counter for creating new Values
valueCntr = 0


def toValues(values):
  return [f"%{v}" for v in values]


def stripQuotes(s):
  return s.replace("\"", "")


# Information regarding the entry fork which is used in the handshake dialect,
# instead of the "Source" nodes in Dynamatic.
entryFork = None


def nextValue():
  global valueCntr
  v = valueCntr
  valueCntr += 1
  return v


class Op:

  def __init__(self):
    self.dotNode = None

    # Type of this operation
    self.type = None
    # A list of integers representing the width of each input operand
    self.ins = []
    # A list of integers representing the width of each output operand
    self.outs = []
    self.name = None
    # Resolved operand names
    self.operands = {}
    # Resolved result names
    self.results = {}
    self.writer = None

  def init(self):
    global valueCntr
    for i, o in enumerate(self.outs):
      self.results[i] = f"{nextValue()}"

    if not self.dotNode:
      return

    # Some commonly used attributes
    if 'op' in self.dotNode[1]:
      self.operator = stripQuotes(self.dotNode[1]['op'])

    if 'value' in self.dotNode[1]:
      self.value = int(stripQuotes(self.dotNode[1]['value']), 16)

  def write(self, writer):
    if len(self.ins) != len(self.operands):
      raise RuntimeError("Number of operands does not match number of ins")

    if len(self.outs) != len(self.results):
      raise RuntimeError("Number of results does not match number of outs")

    self.writer = writer

    keepResults = True
    keepResults &= len(self.outs) != 0

    # Return nodes do not have any results in MLIR
    keepResults &= not (self.type == "Operator" and self.operator == "ret_op")

    if keepResults:
      self.writeResults()
      writer.write(" = ")

    if self.type == "Constant":
      self.writeConstant()
    elif self.type == "Mux":
      self.writeMux()
    elif self.type == "Buffer":
      self.writeBuffer()
    elif self.type == "Source":
      self.writeSource()
    elif self.type == "Branch":
      self.writeBranch()
    elif self.type == "Operator":
      self.writeOperator()
    elif self.isSOSTOp():
      self.writeSOSTOp()
    else:
      self.writeDefaultOp()

    self.writeOpType()
    self.writeInfo()

  def resultName(self, index):
    if index not in self.results:
      raise RuntimeError("Result index not found: " + str(index))
    return self.results[index]

  def operandName(self, index):
    if index not in self.operands:
      raise RuntimeError("Operand index not found: " + str(index))
    return self.operands[index]

  def writeConstant(self):
    self.writer.write(to_handshake_op(self.type))
    self.writer.write(" ")
    self.writeOperands()
    self.writer.write(" {value = " + str(self.value) + " : " +
                      typeResolver.getType(self.results[0]) + "}")

  def writeMux(self):
    self.writer.write(to_handshake_op(self.type))
    self.writer.write(" ")
    self.writeOperands(end=1)
    self.writer.write(" [")
    self.writeOperands(start=1)
    self.writer.write("]")

  def writeBranch(self):
    # Operands are switches in MLIR and dynamatic
    self.writer.write(to_handshake_op(self.type))
    self.writer.write(" ")
    self.writeOperands(1)
    self.writer.write(", ")
    self.writeOperands(0, 1)

  def isSOSTOp(self):
    return self.type in ["Fork", "Join"]

  # Writes a SOST operation (Sized Operation with Single Type).
  # These are operation on the format:
  # opname [size] operands optAttrDict : dataType
  def writeSOSTOp(self):
    if self.type in ["Fork"]:
      n = len(self.outs)
    else:
      n = len(self.ins)

    self.writer.write(to_handshake_op(self.type) + f" [{n}] ")
    self.writeOperands()

  def writeDefaultOp(self):
    self.writer.write(to_handshake_op(self.type))
    if len(self.ins) != 0:
      self.writer.write(" ")
      self.writeOperands()

  def writeBuffer(self):
    self.writer.write(to_handshake_op(self.type))
    # TODO: shouldn't there be an attribute for this in dynamatic IR?
    self.writer.write(" [2] ")
    self.writeOperands()
    self.writer.write(" {sequential = true}")

  def writeSource(self):
    self.writer.write(to_handshake_op(self.type))
    # Source ops in Handshake IR are simple operations with no operands.

  def writeOperator(self):
    self.writer.write(mlir_operator_type(self.operator) + " ")
    self.writeOperands()

  def writeOperands(self, start=None, end=None):
    ops = [self.operands[k] for k in sorted(self.operands.keys())]

    if start is not None:
      ops = ops[start:]
    if end is not None:
      ops = ops[:end]

    self.writer.write(", ".join(toValues(ops)))

  def writeResults(self):
    self.writer.write(", ".join(toValues(self.results.values())))

  def writeOpType(self):
    if self.type == "Source":
      return

    self.writer.write(" : ")

    if self.type == "Mux":
      self.writer.write(typeResolver.getType(self.operands[0]))
      self.writer.write(", ")
      self.writer.write(typeResolver.getType(self.results[0]))
    elif self.type == "CntrlMerge":
      self.writer.write(typeResolver.getType(self.results[0]))
    elif self.type in ["Sink", "Branch"] or self.isSOSTOp():
      self.writer.write(typeResolver.getType(self.operands[0]))
    else:
      if self.type == "Operator":
        if self.operator.startswith("icmp"):
          self.writer.write(typeResolver.getType(self.operands[0]))
          return

      results = [self.results[x] for x in sorted(self.results.keys())]
      types = [typeResolver.getType(x) for x in results]
      self.writer.write(", ".join(types))

  def writeInfo(self):
    if self.dotNode:
      self.writer.write("\t// ")
      self.writer.write(self.dotNode[0])


class EntryFork(Op):

  def __init__(self):
    super().__init__()
    self.type = "Fork"
    self.name = "entryfork"
    self.sourceOutputs = {}

    # The entry fork takes a single none-typed input
    self.ins.append(0)

  def addOutput(self, sourceName):
    self.sourceOutputs[sourceName] = len(self.sourceOutputs)
    # Each output is a none-typed output
    self.outs.append(0)

  def resultName(self, sourceName):
    if sourceName not in self.sourceOutputs:
      raise RuntimeError("Source name not found: " + sourceName)
    return f"%entryfork_{self.sourceOutputs[sourceName]}"


entryFork = EntryFork()


def parseIO(ins):
  ins = ins.split(' ')
  ins.sort()
  ins = [x.strip().replace("\"", "") for x in ins]
  ins = [x for x in ins if x != '']
  ins = [int(i.split(':')[1]) for i in ins]
  return ins


def parseNodes(G):
  ops = {}
  for n in G.nodes(data=True):
    op = Op()
    op.name = stripQuotes(n[0])
    op.type = stripQuotes(n[1]['type'])
    if 'in' in n[1]:
      op.ins = parseIO(stripQuotes(n[1]['in']))
    if 'out' in n[1]:
      op.outs = parseIO(stripQuotes(n[1]['out']))
    op.dotNode = n
    op.init()
    ops[op.name] = op

    # # Each source node is an additional output in the entry fork.
    # if op.type == "Source":
    #   entryFork.addOutput(op.name)

  return ops


def getTrailingNum(s):
  return int(re.compile(r'(\d+)$').search(s).group(1))


def resolveEdges(G, ops):
  for n in G.edges(data=True):
    srcOp = ops[n[0]]
    dstOp = ops[n[1]]

    # Dynamatic is 1 indexed...
    fromAttr = getTrailingNum(stripQuotes(n[2]['from'])) - 1
    toAttr = getTrailingNum(stripQuotes(n[2]['to'])) - 1
    dstOp.operands[toAttr] = srcOp.resultName(fromAttr)


def writeOp(op, writer):
  writer.writeIndent()
  op.write(writer)
  writer.write("\n")


def writeBody(ops, writer):

  def filter(op):
    if op.type in ["Entry", "Exit"]:
      # Irrelevant, these are handled elsewhere
      return True
    if op.type == "Operator" and op.operator == "ret_op":
      return True
    return False

  for op in ops.values():
    if filter(op):
      continue
    writeOp(op, writer)

  # Find return node and write it; this is a terminator operation and must
  # come last.
  for op in ops.values():
    if op.type == "Operator" and op.operator == "ret_op":
      writeOp(op, writer)
      return

  raise RuntimeError("No return node found")


def blockArgFromType(t):
  v = f"%{nextValue()}"
  barg = f"{v} : {toMLIRValueType(t)}"
  return v, barg


# Naively remove any 'subgraph ... {' and a following '}'


def removeSubGraphs(filename):
  with open(filename, 'r') as f:
    lines = f.readlines()

  outLines = []

  bracketCntr = 0
  with open(filename, 'r') as f:
    for line in lines:
      if 'subgraph' in line:
        bracketCntr += 1
        continue
      if '}' in line and bracketCntr > 0:
        bracketCntr -= 1
        continue
      outLines += [line]

  with open(filename, 'w') as f:
    f.writelines(outLines)


def resolveFunctiontypes(nodes):
  # Find entry and exit nodes
  args = []
  results = []
  for entry in [n for n in nodes if stripQuotes(n[1]['type']) == "Entry"]:
    args.append(parseIO(stripQuotes(entry[1]['out']))[0])
  for exit in [n for n in nodes if stripQuotes(n[1]['type']) == "Exit"]:
    results.append(parseIO(stripQuotes(exit[1]['out']))[0])

  return args, results


# Returns the SSA values assigned to the start node outputs, in sorted order.
def getEntryNodeSSAs(ops):
  SSAOps = []
  for n in ops.values():
    if n.type == "Entry":
      SSAOps.append(n)

  # Sort on the names on the start nodes; let's hope that this is in order with the actual function arguments!
  SSAOps.sort(key=lambda x: x.name)

  # Return the SSA values
  return [x.resultName(0) for x in SSAOps]


def getExitNodeSSAs(ops):
  SSAOps = []
  for n in ops.values():
    if n.type == "Exit":
      SSAOps.append(n)

  # Sort on the names on the start nodes; let's hope that this is in order with the actual function arguments!
  SSAOps.sort(key=lambda x: x.name)

  # Return the SSA values
  return [x.operandName(0) for x in SSAOps]


def parseDynamaticFile(fileName, outstream):
  # Remove subgraphs; this is not handled by networkx.
  tmpFile = tempfile.NamedTemporaryFile(delete=False)
  tmpFile.write(open(fileName, 'rb').read())
  removeSubGraphs(tmpFile.name)

  G = nx.DiGraph(read_dot(tmpFile.name))
  basename = os.path.basename(fileName)
  basename = os.path.splitext(basename)[0]
  writer = ModuleWriter(outstream)
  ops = parseNodes(G)
  # We've gathered all node information and can now initialize the entry fork.
  entryFork.init()
  # Add entry ctrl to the entry fork
  entryFork.operands[0] = f"ctrl"
  resolveEdges(G, ops)

  typeResolver.resolveTypes(ops)

  # Go in and figure out which SSA values are used in the
  # start/exit nodes, and use those when printing the arguments and to resolve types.
  funcArgSSAValues = getEntryNodeSSAs(ops)
  funcResSSAValues = getExitNodeSSAs(ops)

  funcArgs = [
      f"%{ssa} : {typeResolver.getType(ssa)}" for ssa in funcArgSSAValues
  ]
  funcResults = [f"{typeResolver.getType(ssa)}" for ssa in funcResSSAValues]
  functypestr = "(" + ", ".join(funcArgs) + ") -> (" + ", ".join(
      funcResults) + ")"
  writer.writeRegion(
      "module",
      lambda: writer.writeRegion(f"handshake.func @{basename} " + functypestr,
                                 lambda: writeBody(ops, writer)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Parse a Dynamatic file')
  parser.add_argument('file', help='Dynamatic file to parse')
  parser.add_argument('-o',
                      '--output',
                      help='Output file',
                      type=str,
                      default=None)
  args = parser.parse_args()

  # Create a stream object
  if args.output:
    outstream = open(args.output, 'w')
  else:
    outstream = sys.stdout

  parseDynamaticFile(args.file, outstream)
