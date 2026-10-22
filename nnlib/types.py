"""Array type aliases used throughout the course.

Type hints cannot check shapes, so every array in the course also carries a
shape comment: `X: FloatArray  # (n, d_in)`.
"""

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]
