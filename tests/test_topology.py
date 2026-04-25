import numpy as np
import pytest

from celnn.core.exceptions import BackendError
from celnn.core.topology import RegularGridTopology


def test_identity_template_has_center_weight():
    topology = RegularGridTopology(shape=(8, 8), boundary="reflect")
    template = topology.identity_template()
    assert template.shape == (3, 3)
    assert np.isclose(template[1, 1], 1.0)
    assert np.isclose(template.sum(), 1.0)


def test_numpy_fallback_support_is_limited_to_1d_and_2d():
    topology = RegularGridTopology(shape=(4, 4, 4))
    assert topology.numpy_fallback_supported() is False
    with pytest.raises(BackendError):
        topology.require_numpy_fallback_support()
