# Copyright 2026 The TensorNetwork-NG Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
from tensornetwork.backends.numpy import decompositions

jax.config.update("jax_enable_x64", True)


def test_qr_non_negative_diagonal_complex():
    matrix = jnp.asarray(np.array([[0.0, 1 + 2j], [0.0, 3 - 1j]]))
    q, r = decompositions.qr(jnp, matrix, 1, True)
    np.testing.assert_allclose(np.asarray(q @ r), np.asarray(matrix))
    diag = np.diagonal(np.asarray(r))
    np.testing.assert_allclose(diag.imag, np.zeros(2))
    assert np.all(diag.real >= 0)


def test_rq_non_negative_diagonal_complex():
    matrix = jnp.asarray(np.array([[0.0, 0.0], [1 + 2j, 3 - 1j]]))
    r, q = decompositions.rq(jnp, matrix, 1, True)
    np.testing.assert_allclose(np.asarray(r @ q), np.asarray(matrix))
    diag = np.diagonal(np.asarray(r))
    np.testing.assert_allclose(diag.imag, np.zeros(2))
    assert np.all(diag.real >= 0)
