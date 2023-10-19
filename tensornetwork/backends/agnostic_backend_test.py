from functools import partial
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import tensorflow as tf


@pytest.mark.parametrize("K", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
def test_backend_methods(K):
    np.testing.assert_allclose(
        K.prod(K.convert_to_tensor(np.ones([3, 2, 5])), axis=[1]), np.ones([3, 5])
    )
