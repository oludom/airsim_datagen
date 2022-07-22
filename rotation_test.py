import numpy as np

from util import *
from math import degrees, radians

by = 45
by = radians(by)
b = np.array([1, 1, 0])
v = np.array([1, 2, 0])

bmv = b - v

inbf = vector_world_to_body(v, b, by)
inwf = vector_body_to_world(inbf, b, by)

print(f"inbf: {inbf}, inwf: {inwf}")
