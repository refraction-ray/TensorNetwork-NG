"""
directlt copied from linalg submodule
"""

from typing import Any, Optional, Text, List
from tensornetwork.network_components import Node

AbstractNode = Any


def conj(
    node: AbstractNode,
    name: Optional[Text] = None,
    axis_names: Optional[List[Text]] = None,
) -> AbstractNode:
    """Conjugate a `node`.

    Args:
      node: A `AbstractNode`.
      name: Optional name to give the new node.
      axis_names: Optional list of names for the axis.

    Returns:
      A new node. The complex conjugate of `node`.

    Raises:
      AttributeError: If `node` has no `backend` attribute.
    """
    if not hasattr(node, "backend"):
        raise AttributeError(
            "Node {} of type {} has no `backend`".format(node, type(node))
        )
    backend = node.backend
    if not axis_names:
        axis_names = node.axis_names

    return Node(
        backend.conj(node.tensor), name=name, axis_names=axis_names, backend=backend
    )
