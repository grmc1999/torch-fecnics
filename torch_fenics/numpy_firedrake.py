#import fenics
import firedrake
from  firedrake import adjoint
#import fenics_adjoint
import numpy as np

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch

import firedrake as fd
from firedrake import adjoint



def firedrake_to_numpy(obj: Any, *, copy: bool = True) -> np.ndarray:
    """
    Convert Firedrake objects (Function, Constant, adjoint.AdjFloat) -> numpy array.

    Notes:
      - This returns LOCAL data on each MPI rank (no gather).
      - For Functions, uses .dat.data_ro (no halos).
    """
    # Firedrake Constant
    if isinstance(obj, fd.Constant):
        # Constant.values() exists in Firedrake; returns scalar or array-like.
        arr = np.asarray(obj.values())
        return arr.copy() if copy else arr

    # Firedrake Function
    if isinstance(obj, fd.Function):
        arr = obj.dat.data_ro  # owned dofs, no halos
        out = np.asarray(arr)
        return out.copy() if copy else out

    # adjoint scalar (pyadjoint.AdjFloat underneath)
    if hasattr(adjoint, "AdjFloat") and isinstance(obj, adjoint.AdjFloat):
        return np.array(float(obj), dtype=np.float64)

    # Plain python numeric
    if np.isscalar(obj):
        return np.array(obj, dtype=np.float64)

    raise TypeError(f"Cannot convert type {type(obj)} to numpy.")


def numpy_to_firedrake(a: np.ndarray, template: Any) -> Any:
    """
    Convert numpy array -> Firedrake object shaped like `template`.

    Supported templates:
      - fd.Constant
      - fd.Function
      - adjoint.AdjFloat (or scalar)
    """
    a = np.asarray(a)

    # Firedrake Constant: create a new Constant with same value shape
    if isinstance(template, fd.Constant):
        # Firedrake Constant constructor accepts scalar or array-like
        return fd.Constant(a.item() if a.shape == () else a)

    # Firedrake Function: create a new Function in same space and assign local dofs
    if isinstance(template, fd.Function):
        V = template.function_space()
        u = fd.Function(V)

        target = u.dat.data  # writable view, owned dofs (no halos)
        target_dtype = target.dtype

        aa = np.asarray(a, dtype=target_dtype)

        # Match shape exactly:
        # - scalar space: (ndofs,)
        # - vector/tensor space: (ndofs, value_size, ...)
        if aa.shape != target.shape:
            # allow flat input and reshape if total size matches
            if aa.size != target.size:
                raise ValueError(
                    f"Wrong numpy size for Function assignment: got shape {aa.shape} "
                    f"(size={aa.size}) expected {target.shape} (size={target.size})."
                )
            aa = aa.reshape(target.shape)

        target[:] = aa
        return u

    # adjoint scalar (AdjFloat) or numeric
    if hasattr(adjoint, "AdjFloat") and isinstance(template, adjoint.AdjFloat):
        return adjoint.AdjFloat(float(a))

    if np.isscalar(template):
        return float(a)

    raise TypeError(f"Cannot convert numpy array to template type {type(template)}.")