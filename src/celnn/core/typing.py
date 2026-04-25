"""Common typing aliases for celnn."""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike
NDArray = np.ndarray[Any, np.dtype[Any]]
ActivationLike = str | Callable[[NDArray], NDArray]
BoundaryMode = Literal["constant", "wrap", "reflect", "nearest", "mirror"]
MetadataDict = dict[str, Any]
