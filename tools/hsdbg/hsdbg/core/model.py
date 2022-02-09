class CallbackTrait:

  def __init__(self, callback=None):
    self.callback = callback

  def setCallback(self, callback):
    self.callback = callback

  def __call__(self, *args, **kwargs):
    return self.callback(*args, **kwargs)


class Model:
  """ Top-level class representing an instance of a design/simulation.
  """

  def __init__(self) -> None:
    self.timestep = 1
    self.step = 0
    self.trace = None

  def getTrace(self):
    """Returns the trace object associated with this model.
    """
    if not self.trace:
      raise Exception("Trace not set!")
    return self.trace

  def beginTime(self):
    """Returns the first timestep of the simulation.
    """
    return self.trace.getStartTime()

  def endTime(self):
    """Returns the last timestep of the simulation.
    """
    return self.trace.getEndTime()

  def currentStep(self):
    """Returns the current simulation step.
    """
    return self.step

  def setStep(self, step: int):
    """Updates the model to the given simulation step.
    """

    if step < self.beginTime():
      print(f"Capping at minimum time ({self.beginTime()})")
      step = self.beginTime()
    elif step > self.endTime():
      print(f"Capping at maximum time ({self.endTime()})")
      step = self.endTime()
    self.step = step

    self.updateModel()
