import argparse
from vcdvcd import VCDVCD
import tempfile
import os
import subprocess
from hsdbg.frontends.dotimageserver import DotImageServer
from hsdbg.core.utils import *
"""HSDbg
HSDbg is a script which couples together a .dot representation of a CIRCT
handshake program, and a .vcd trace of a simulation on the same program.
This is done by correlating the .vcd hierarchy with the DOT hierarchy, and
correlating the VCD signals with what looks to be the same values in the DOT
file. Since .dot vertices are created on the handshake operation level,
the script relies on additional attributes in the .dot file to disambiguated
which in- and output edges of a node corresponds to which signal in the .vcd.

.dot files are rendered to .svg images and served by a web server. This server
runs a simple web page which will automatically update the shown image.
"""

from hsdbg.frontends.handshake.handshake import *

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="HSDBG: Visual simulation of domain-specific IR",)

  subparsers = parser.add_subparsers(dest="mode", help="Mode option")

  # Initialize targets
  targets = {}

  def addTarget(type):
    subparser = subparsers.add_parser(type.name())
    type.addArguments(subparser)
    targets[type.name()] = type

  addTarget(HandshakeModel)

  # Parse args
  args = parser.parse_args()

  if args.mode not in targets:
    print("Invalid mode: {}".format(args.mode))
    parser.print_help()
    exit(1)

  # Run target
  targets[args.mode](args)
