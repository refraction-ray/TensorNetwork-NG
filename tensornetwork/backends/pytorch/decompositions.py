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
"""Tensor Decomposition Implementations."""

from typing import Optional, Tuple, Any
import numpy as np

Tensor = Any


def _prod(iterable):
    result = 1
    for dim in iterable:
        result = result * dim
    return result


def svd(
    torch: Any,
    tensor: Tensor,
    pivot_axis: int,
    max_singular_values: Optional[int] = None,
    max_truncation_error: Optional[float] = None,
    relative: Optional[bool] = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the singular value decomposition (SVD) of a tensor.

    The SVD is performed by treating the tensor as a matrix, with an effective
    left (row) index resulting from combining the axes `tensor.shape[:pivot_axis]`
    and an effective right (column) index resulting from combining the axes
    `tensor.shape[pivot_axis:]`.

    For example, if `tensor` had a shape (2, 3, 4, 5) and `pivot_axis` was 2, then
    `u` would have shape (2, 3, 6), `s` would have shape (6), and `vh` would
    have shape (6, 4, 5).

    If `max_singular_values` is set to an integer, the SVD is truncated to keep
    at most this many singular values.

    If `max_truncation_error > 0`, as many singular values will be truncated as
    possible, so that the truncation error (the norm of discarded singular
    values) is at most `max_truncation_error`.
    If `relative` is set `True` then `max_truncation_err` is understood
    relative to the largest singular value.

    If both `max_singular_values` snd `max_truncation_error` are specified, the
    number of retained singular values will be
    `min(max_singular_values, nsv_auto_trunc)`, where `nsv_auto_trunc` is the
    number of singular values that must be kept to maintain a truncation error
    smaller than `max_truncation_error`.

    In the error-truncation mode (`max_truncation_error` set), the
    number of retained singular values is derived from the data.
    This mode is incompatible with `torch.func.vmap`. Full-graph
    gradient compilation requires setting
    `torch._dynamo.config.capture_scalar_outputs = True` and
    torch >= 2.8, for losses that depend only on the singular
    values.

    In the fixed-count mode (`max_truncation_error = None` with an
    explicit `max_singular_values`), the decomposition traces under
    `torch.func.vmap`. Its gradients compile without extra
    configuration since torch 2.4, for losses that depend only on
    the singular values. On torch 2.1-2.3, they additionally
    require `torch._dynamo.config.capture_func_transforms = True`.

    For complex inputs, full-graph compilation of gradients of
    losses using `u` or `vh` additionally requires torch >= 2.12,
    in either mode. Both modes, and gradients of losses that
    depend only on the singular values, work eagerly without any
    compilation settings.

    The output consists of three tensors `u, s, vh` such that:
    ```python
      u[i1,...,iN, j] * s[j] * vh[j, k1,...,kM] == tensor[i1,...,iN, k1,...,kM]
    ```
    Note that the output ordering matches numpy.linalg.svd rather than tf.svd.

    Args:
      tf: The tensorflow module.
      tensor: A tensor to be decomposed.
      pivot_axis: Where to split the tensor's axes before flattening into a
        matrix.
      max_singular_values: The number of singular values to keep, or `None` to
        keep them all.
      max_truncation_error: The maximum allowed truncation error or `None` to not
        do any truncation.
      relative: Multiply `max_truncation_err` with the largest singular value.

    Returns:
      u: Left tensor factor.
      s: Vector of ordered singular values from largest to smallest.
      vh: Right tensor factor.
      s_rest: Vector of discarded singular values (length zero if no
              truncation).
    """
    left_dims = list(tensor.shape)[:pivot_axis]
    right_dims = list(tensor.shape)[pivot_axis:]

    tensor = torch.reshape(tensor, (_prod(left_dims), _prod(right_dims)))
    u, s, vh = torch.linalg.svd(tensor, full_matrices=False)

    if max_singular_values is None:
        max_singular_values = s.nelement()

    if max_truncation_error is not None:
        # Cumulative norms of singular values in ascending order
        s_sorted, _ = torch.sort(s**2)
        trunc_errs = torch.sqrt(torch.cumsum(s_sorted, 0))
        # If relative is true, rescale max_truncation error with the largest
        # singular value to yield the absolute maximal truncation error.
        if relative:
            abs_max_truncation_error = max_truncation_error * s[0]
        else:
            abs_max_truncation_error = max_truncation_error
        # We must keep at least this many singular values to ensure the
        # truncation error is <= abs_max_truncation_error.
        num_sing_vals_keep = (
            torch.sum(trunc_errs > abs_max_truncation_error)
            .clamp(max=max_singular_values)
            .clamp(max=s.nelement())
            .item()
        )

        check_fn = getattr(torch, "_check", None)
        if check_fn is not None:
            check_fn(num_sing_vals_keep >= 0)
            check_fn(num_sing_vals_keep <= s.nelement())
    else:
        num_sing_vals_keep = min(max_singular_values, s.nelement())

    # we recast to the original dtype.
    s = s.to(tensor.dtype)

    s_rest = s[num_sing_vals_keep:]
    s = s[:num_sing_vals_keep]
    u = u[:, :num_sing_vals_keep]
    vh = vh[:num_sing_vals_keep]

    dim_s = s.shape[0]
    u = torch.reshape(u, left_dims + [dim_s])
    vh = torch.reshape(vh, [dim_s] + right_dims)

    return u, s, vh, s_rest


def qr(
    torch: Any, tensor: Tensor, pivot_axis: int, non_negative_diagonal: bool = False
) -> Tuple[Tensor, Tensor]:
    """Computes the QR decomposition of a tensor.

    The QR decomposition is performed by treating the tensor as a matrix,
    with an effective left (row) index resulting from combining the axes
    `tensor.shape[:pivot_axis]` and an effective right (column) index
    resulting from combining the axes `tensor.shape[pivot_axis:]`.

    For example, if `tensor` had a shape (2, 3, 4, 5) and `pivot_axis` was 2,
    then `q` would have shape (2, 3, 6), and `r` would have shape (6, 4, 5).

    The output consists of two tensors `Q, R` such that:
    ```python
      Q[i1,...,iN, j] * R[j, k1,...,kM] == tensor[i1,...,iN, k1,...,kM]
    ```
    `R` is an upper triangular matrix, `Q` is an orthonormal matrix
    Note that the output ordering matches numpy.linalg.svd rather than tf.svd.

    Args:
      tf: The tensorflow module.
      tensor: A tensor to be decomposed.
      pivot_axis: Where to split the tensor's axes before flattening into a
        matrix.

    Returns:
      Q: Left tensor factor.
      R: Right tensor factor.
    """

    left_dims = list(tensor.shape)[:pivot_axis]
    right_dims = list(tensor.shape)[pivot_axis:]

    tensor = torch.reshape(tensor, (np.prod(left_dims), np.prod(right_dims)))
    q, r = torch.linalg.qr(tensor)
    if non_negative_diagonal:
        phases = torch.sgn(torch.diagonal(r))
        phases = torch.where(phases == 0, torch.ones_like(phases), phases)
        q = q * phases
        r = torch.conj(phases)[:, None] * r
    center_dim = q.shape[1]
    q = torch.reshape(q, list(left_dims) + [center_dim])
    r = torch.reshape(r, [center_dim] + list(right_dims))
    return q, r


def rq(
    torch: Any, tensor: Tensor, pivot_axis: int, non_negative_diagonal: bool = False
) -> Tuple[Tensor, Tensor]:
    """Computes the RQ decomposition of a tensor.

    The RQ decomposition is performed by treating the tensor as a matrix,
    with an effective left (row) index resulting from combining the axes
    `tensor.shape[:pivot_axis]` and an effective right (column) index
    resulting from combining the axes `tensor.shape[pivot_axis:]`.

    For example, if `tensor` had a shape (2, 3, 4, 5) and `pivot_axis` was 2,
    then `r` would have shape (2, 3, 6), and `q` would have shape (6, 4, 5).

    The output consists of two tensors `R, Q` such that:
    ```python
      R[i1,...,iN, j] * Q[j, k1,...,kM] == tensor[i1,...,iN, k1,...,kM]
    ```
    `R` is a lower triangular matrix, `Q` is an orthonormal matrix
    Note that the output ordering matches numpy.linalg.svd rather than tf.svd.

    Args:
      tf: The tensorflow module.
      tensor: A tensor to be decomposed.
      pivot_axis: Where to split the tensor's axes before flattening into a
       p matrix.

    Returns:
      R: Left tensor factor.
      Q: Right tensor factor.
    """

    left_dims = tensor.shape[:pivot_axis]
    right_dims = tensor.shape[pivot_axis:]
    tensor = torch.reshape(tensor, [np.prod(left_dims), np.prod(right_dims)])
    q, r = torch.linalg.qr(torch.transpose(tensor, 0, 1))
    if non_negative_diagonal:
        phases = torch.sgn(torch.diagonal(r))
        phases = torch.where(phases == 0, torch.ones_like(phases), phases)
        q = q * phases
        r = torch.conj(phases)[:, None] * r
    r, q = torch.transpose(r, 0, 1), torch.transpose(q, 0, 1)  # M=r*q at this point
    center_dim = r.shape[1]
    r = torch.reshape(r, list(left_dims) + [center_dim])
    q = torch.reshape(q, [center_dim] + list(right_dims))
    return r, q
