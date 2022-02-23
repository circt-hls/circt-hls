from hsdbg.frontends.dotfile import *
from hsdbg.core.utils import *
from hsdbg.core.model import *
from hsdbg.frontends.dotimageserver import *

import subprocess


class DotModelEdge():

  def __init__(self) -> None:
    super().__init__()

    # Handle to a DotEdge object, see dotfile.py
    self.edge = None


class DotModelNode():

  def __init__(self) -> None:
    super().__init__()


class DotModel(Model):
  """ A model which parses and manipulates a dot file to create a visualization
  of the target model.
  """

  @staticmethod
  def addArguments(subparser):
    subparser.add_argument("--dot", help="The dotfile to use.", type=str)
    subparser.add_argument(
        "-f",
        "--format",
        help="The value format of data signals. options are 'hex, bin, dec'.",
        type=str,
        default="dec")
    subparser.add_argument(
        "-p",
        "--port",
        help="The port to run the image server on. default='8080'",
        type=int,
        default=8080)

  def __init__(self, dot, port) -> None:
    super().__init__()

    if not dot:
      raise ValueError("No dot file specified.")

    # Maintain the source code of the dot file. Since there aren't any good
    # python libraries for in-place modification of
    self.dotFile = DotFile(dot)
    self.workingImagePath = self.setupWorkingImage(dot)

    # Ensure the initial state of the working image file.
    self.dotUpdateFinished()

    # Port to run the image server on.
    self.port = port

  def dotUpdateFinished(self):
    # Dump the modified dot file to a temporary file.
    workingDotfile = self.workingImagePath + ".dot"
    self.dotFile.dump(workingDotfile)

    # Run 'dot' on the file to produce an svg file and place it in the
    # working image path, updating any viewers that have the file opened.
    subprocess.call(
        ["dot", workingDotfile, "-Tsvg", "-o", self.workingImagePath])

    # Reset modifications to original dot file after each update.
    self.dotFile.reset()

  def setupWorkingImage(self, dotFile):
    temp_dir = tempfile.gettempdir()
    dotFileName = dotFile.split("/")[-1]
    temp_path = os.path.join(temp_dir, dotFileName + ".svg")
    return temp_path

  def startImageServer(self):
    viewer = DotImageServer(self, self.port)
    viewer.start()
    start_interactive_mode(self.port, self)
