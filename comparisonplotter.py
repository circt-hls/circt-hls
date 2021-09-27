from pathlib import Path
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

COMP_DIR = Path.cwd() / "out" / "comparisons"

def prettifyResourceName(resourceName):
    mapping = defaultdict(lambda x: x)
    mapping.update({
        "lut" : "LUT",
        "clb_registers" : "CLB Registers"
    })
    return mapping[resourceName]

def prettifyTestName(testName):
    testName = testName.capitalize()
    testName.replace("_", " ")
    return testName

def fatalError(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)

class CompPlotter():
    def __init__(self, datasets):
        self.datasets = datasets

    def plot(self):
        return


def getRunResource(test, runName, resourceName):
    """ Returns the resource file from runName as a dictionary"""
    fn = COMP_DIR / test / runName / "resources.json"
    with open(fn) as f:
        d = json.load(f)
        if resourceName not in d:
            fatalError(f"requested resource {resourceName} not found in {fn}")
        return d[resourceName]

def getTestResults(test, resourceName, runs, normalizeTo=None):
    resources = {run : getRunResource(test, run, resourceName) for run in runs}
    if normalizeTo != None:
        for key, val in resources.items():
            if key != normalizeTo:
                resources[key] = resources[key]/resources[normalizeTo]
        resources[normalizeTo] = 1.0
    return resources


def plotTests(tests, runs, resourceName, normalizeTo):
    results = {}
    NTests = len(tests)
    NRuns = len(runs)

    # Gather resource results for each test
    for test in tests:
        results[test] = getTestResults(test, resourceName, runs, normalizeTo)
    


    # Unpack results into a list for each run type
    resultsUnpacked = defaultdict(list)
    for test in tests:
        for key, val in results[test].items():
            resultsUnpacked[key].append(val)

    X = np.arange(NTests)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    d = 1.0/NRuns

    patterns = [ "**", "||" , "\\\\" , "//" , "++" , "--", "..","xx", "oo", "OO" ]
    for i, run in enumerate(runs):
        xPos = X + i*d - (1.0 / NRuns)
        yVals = resultsUnpacked[run]
        ax.bar(xPos, yVals,
            width = d,
            color='white',
            edgecolor='black',
            hatch=patterns[i]
        )

        for j, val in enumerate(yVals):
            ax.text(xPos[j], val, str(round(val, 2)),
                horizontalalignment='center',
                verticalalignment='bottom'
            )

    ax.legend(labels=runs)
    plt.xticks(X, [prettifyTestName(test) for test in tests])
    plt.title(prettifyResourceName(resourceName))
    plt.ylabel("Norm. (Vivado)")
    plt.xlabel("Test")
    plt.tight_layout()
    plt.show()


def getRunNames(args):
    """
    Returns the run name for the given argument. 
    These are identical to the names generated in the out/comparisons/... directory.
    """
    runs = []
    def appendIfExists(flag, val):
        if flag:
            runs.append(val)

    appendIfExists(args.circt_dynamic, "circt-dynamic")
    appendIfExists(args.circt_static, "circt-static")
    appendIfExists(args.vivado_hls, "vivado-hls")
    return runs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CIRCT HLS comparison plotter"
    )

    parser.add_argument(
        "tests", type=str, help="Tests to plot, comma separated")
    parser.add_argument(
        "--resources", type=str, help="resources to plot (comma separated). "
        "Valid keys are everything which is present in the output "
        "resources.json file from a compilation run.", required=True)
    parser.add_argument(
        "--vivado-hls", help="Plots Vivado HLS", action="store_true")
    parser.add_argument(
        "--circt-dynamic", help="Plots the CIRCT-based dynamically scheduled flow", action="store_true")
    parser.add_argument(
        "--circt-static", help="Plots the CIRCT-based statically scheduled flow", action="store_true")
    parser.add_argument(
        "--normalize", type=str, help="Normalizes resource usages relative to the provided run")

    args = parser.parse_args()

    # Ensure that the requested tests are available
    tests = args.tests.split(",")
    for test in tests:
        testdir = Path.cwd() / "out" / "comparisons" / test
        if not testdir.exists() or not testdir.is_dir():
            print(f"Found no directory named {testdir}")
            sys.exit(1)

    runs = getRunNames(args)
    if len(runs) == 0:
        fatalError("Must specify at least one run type to plot")
        sys.exit(1)
    
    if args.normalize and args.normalize not in runs:
        fatalError(f"Requested normalization wrt. run '{args.normalize}', but no run exists with that name")

    resources = args.resources.split(",")
    for resource in resources:
        plotTests(tests, runs, resource, args.normalize)
