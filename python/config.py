import os
import glob
import sys

## Import cpp pybind modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.pyd"), recursive=True)]

## COMMON VALUES

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

SAVE_ROOT_PATH = "data"

MODEL_SIZE = 0.97
VOXEL_BNDS = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
VOXEL_SIZE = 0.01

NUMBER_MODEL_SAMPLE_POINTS = 3000000

OCCUPIED_PROB = 0.7
FREE_PROB = 0.4
OCCUPIED_RENDER_PROB = 0.65
UNKNOWN_PROB = 0.5

RADIUS = 3
# RADIUSES = [2, 2.5, 3.0, 3.5, 4.0]
RADIUSES = [3]
ROTATION_STEP = 18
ELEVATATION_NUM = 8
ELEVATATION_STEP = 10

MAX_ITERATE = 20

SURFACE_THRESHOLD = 0.002

POLICY = ["occlusion_aware",
          "unobserved_voxel",
          "rear_side_voxel",
          "rear_side_entropy",
          "proposed_method"]