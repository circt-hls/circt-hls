import logging
from flask import Flask, render_template, send_file, jsonify
from threading import Thread
import readchar


class DotImageServer:
  # An image server which can be used to display the current state of the dot model
  # graph.

  def __init__(self, dotmodel, port):
    self.dotmodel = dotmodel
    self.port = port
    self.server = None

  def finish(self):
    return

  def start(self):
    """ Starts the image server in a separate thread."""

    def runServer():
      # A simple flask application which serves index.html to continuously update
      # the svg file.
      log = logging.getLogger('werkzeug')
      log.setLevel(logging.ERROR)
      app = Flask(__name__)
      app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

      @app.route("/")
      def index():
        return render_template("index.html")

      @app.route("/step")
      def step():
        return jsonify(step=self.dotmodel.currentStep())

      # Serve the svg file on any path request.
      @app.route("/<path:path>")
      def serveFile(path):
        return send_file(self.dotmodel.workingImagePath,
                         mimetype="image/svg+xml")

      app.run(host="localhost", port=self.port)

    self.server = Thread(target=lambda: runServer())
    self.server.start()


def start_interactive_mode(port, model):
  msg = f"""=== Handshake interactive simulation viewer ===

Usage:
    Open "http://localhost:{port}" in a browser to view the simulation.

    right arrow: step forward 1 timestep in vcd time
    left arrow: step backwards 1 timestep in vcd time
    g <step>: step to a specific step in vcd time

"Entering interactive mode. Type 'q' to quit."
"""
  print(flush=True)
  print(msg)
  while True:
    print(f"\r> [{model.currentStep()}] ", end="")
    command = readchar.readkey()
    if command == "q":
      print("Exiting...")
      break
    elif command == '\x1b[D':
      model.setStep(model.currentStep() - 1)
    elif command == '\x1b[C':
      model.setStep(model.currentStep() + 1)
    elif command == "g":
      step = input("Goto step: ")
      model.setStep(step)
