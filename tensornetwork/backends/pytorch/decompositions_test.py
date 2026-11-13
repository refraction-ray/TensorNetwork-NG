# Copyright 2019 The TensorNetwork Authors
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

import math
import numpy as np
import pytest
import torch
from tensornetwork.backends.pytorch import decompositions


def test_expected_shapes():
    val = torch.zeros((2, 3, 4, 5))
    u, s, vh, _ = decompositions.svd(torch, val, 2)
    assert u.shape == (2, 3, 6)
    assert s.shape == (6,)
    np.testing.assert_allclose(s, np.zeros(6))
    assert vh.shape == (6, 4, 5)


def test_svd_complex_reconstruction_gradient():
    val = torch.tensor(
        [[1 + 2j, 3 - 1j, 2 + 0.5j], [-2 + 1j, 0.3 - 2j, 1 + 1j]],
        dtype=torch.complex128,
        requires_grad=True,
    )

    def loss(matrix):
        u, s, vh, _ = decompositions.svd(torch, matrix, 1)
        return torch.real((u * s).mm(vh)).square().sum()

    assert torch.autograd.gradcheck(loss, (val,))


@pytest.mark.parametrize(
    "shape, pivot_axis, dynamic", [((6, 4), 1, False), ((2, 3, 4), 2, True)]
)
def test_svd_compile_grad_singular_values(shape, pivot_axis, dynamic):
    torch.manual_seed(42)
    val = torch.randn(*shape, dtype=torch.complex128)

    def loss(matrix):
        _, s, _, _ = decompositions.svd(
            torch, matrix, pivot_axis, max_singular_values=3
        )
        return (s**2).sum().real

    expected = torch.func.grad(loss)(val)
    actual = torch.compile(
        torch.func.grad(loss), backend="eager", dynamic=dynamic, fullgraph=True
    )(val)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_svd_vmap_grad_singular_values():
    torch.manual_seed(43)
    val = torch.randn(3, 6, 4, dtype=torch.complex128)

    def loss(matrix):
        _, s, _, _ = decompositions.svd(torch, matrix, 1, max_singular_values=3)
        return (s**2).sum().real

    actual = torch.func.vmap(torch.func.grad(loss))(val)
    expected = torch.func.grad(loss)(val[1])
    np.testing.assert_allclose(actual[1], expected, rtol=1e-6, atol=1e-8)


def test_svd_compile_grad_truncation_error():
    torch.manual_seed(44)
    val = torch.randn(6, 4, dtype=torch.complex128)

    def loss(matrix):
        _, s, _, _ = decompositions.svd(torch, matrix, 1, max_truncation_error=2.0)
        return (s**2).sum().real

    expected = torch.func.grad(loss)(val)
    previous = torch._dynamo.config.capture_scalar_outputs
    torch._dynamo.config.capture_scalar_outputs = True
    try:
        actual = torch.compile(torch.func.grad(loss), backend="eager", fullgraph=True)(
            val
        )
    finally:
        torch._dynamo.config.capture_scalar_outputs = previous
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_expected_shapes_qr():
    val = torch.zeros((2, 3, 4, 5))
    for non_negative_diagonal in [True, False]:
        q, r = decompositions.qr(torch, val, 2, non_negative_diagonal)
        assert q.shape == (2, 3, 6)
        assert r.shape == (6, 4, 5)


def test_expected_shapes_rq():
    val = torch.zeros((2, 3, 4, 5))
    for non_negative_diagonal in [True, False]:
        r, q = decompositions.rq(torch, val, 2, non_negative_diagonal)
        assert r.shape == (2, 3, 6)
        assert q.shape == (6, 4, 5)


def test_rq():
    random_matrix = torch.rand([10, 10], dtype=torch.float64)
    for non_negative_diagonal in [True, False]:
        r, q = decompositions.rq(torch, random_matrix, 1, non_negative_diagonal)
        np.testing.assert_allclose(r.mm(q), random_matrix)


def test_qr():
    random_matrix = torch.rand([10, 10], dtype=torch.float64)
    for non_negative_diagonal in [True, False]:
        q, r = decompositions.rq(torch, random_matrix, 1, non_negative_diagonal)
        np.testing.assert_allclose(q.mm(r), random_matrix)


def test_qr_complex_reconstruction():
    matrix = torch.tensor([[0.0, 1 + 2j], [0.0, 3 - 1j]], dtype=torch.complex128)
    for non_negative_diagonal in [True, False]:
        q, r = decompositions.qr(torch, matrix, 1, non_negative_diagonal)
        np.testing.assert_allclose(q.mm(r), matrix)
        if non_negative_diagonal:
            diag = torch.diagonal(r)
            assert torch.all(diag.imag.abs() < 1e-10)
            assert torch.all(diag.real >= 0)


def test_rq_complex_reconstruction():
    matrix = torch.tensor([[0.0, 0.0], [1 + 2j, 3 - 1j]], dtype=torch.complex128)
    for non_negative_diagonal in [True, False]:
        r, q = decompositions.rq(torch, matrix, 1, non_negative_diagonal)
        np.testing.assert_allclose(r.mm(q), matrix)
        if non_negative_diagonal:
            diag = torch.diagonal(r)
            assert torch.all(diag.imag.abs() < 1e-10)
            assert torch.all(diag.real >= 0)


def test_max_singular_values():
    np.random.seed(2018)
    random_matrix = np.random.rand(10, 10)
    unitary1, _, unitary2 = np.linalg.svd(random_matrix)
    singular_values = np.array(range(10))
    val = unitary1.dot(np.diag(singular_values).dot(unitary2.T))
    u, s, vh, trun = decompositions.svd(
        torch, torch.tensor(val), 1, max_singular_values=7
    )
    assert u.shape == (10, 7)
    assert s.shape == (7,)
    np.testing.assert_array_almost_equal(s, np.arange(9, 2, -1))
    assert vh.shape == (7, 10)
    np.testing.assert_array_almost_equal(trun, np.arange(2, -1, -1))


def test_max_truncation_error():
    np.random.seed(2019)
    random_matrix = np.random.rand(10, 10)
    unitary1, _, unitary2 = np.linalg.svd(random_matrix)
    singular_values = np.array(range(10))
    val = unitary1.dot(np.diag(singular_values).dot(unitary2.T))
    u, s, vh, trun = decompositions.svd(
        torch, torch.Tensor(val), 1, max_truncation_error=math.sqrt(5.1)
    )
    assert u.shape == (10, 7)
    assert s.shape == (7,)
    np.testing.assert_array_almost_equal(s, np.arange(9, 2, -1), decimal=5)
    assert vh.shape == (7, 10)
    np.testing.assert_array_almost_equal(trun, np.arange(2, -1, -1))


def test_max_truncation_error_relative():
    absolute = np.diag([2.0, 1.0, 0.2, 0.1])
    relative = np.diag([2.0, 1.0, 0.2, 0.1])
    max_truncation_err = 0.2
    _, _, _, trunc_sv_absolute = decompositions.svd(
        torch,
        torch.Tensor(absolute),
        1,
        max_truncation_error=max_truncation_err,
        relative=False,
    )
    _, _, _, trunc_sv_relative = decompositions.svd(
        torch,
        torch.Tensor(relative),
        1,
        max_truncation_error=max_truncation_err,
        relative=True,
    )
    np.testing.assert_almost_equal(trunc_sv_absolute, [0.1])
    np.testing.assert_almost_equal(trunc_sv_relative, [0.2, 0.1])
