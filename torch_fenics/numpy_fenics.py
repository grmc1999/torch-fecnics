from dolfinx import fem
import numpy as np

import pyadjoint  # keep if you really use AdjFloat/pyadjoint objects

try:
    from petsc4py import PETSc
except ImportError:
    PETSc = None


def _owned_size(V: fem.FunctionSpace):
    """Number of owned (non-ghost) entries in the dof vector for this space."""
    imap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs  # block size (e.g. vector-valued spaces)
    return imap.size_local * bs, bs


def fenics_to_numpy(obj, *, owned_only: bool = True, reshape_blocks: bool = False, copy: bool = True):
    """
    Convert DOLFINx objects to numpy arrays.

    Notes:
      - Returns LOCAL data on each MPI rank (not gathered).
      - If owned_only=True, ghost entries are excluded.
      - If reshape_blocks=True and block-size > 1, reshapes to (-1, bs).
    """
    # DOLFINx Constant
    if isinstance(obj, fem.Constant):
        arr = np.asarray(obj.value)
        return arr.copy() if copy else arr

    # DOLFINx Function
    if isinstance(obj, fem.Function):
        # Make sure ghost values are consistent before reading
        obj.x.scatter_forward()

        V = obj.function_space
        n_owned, bs = _owned_size(V)

        x = obj.x.array  # includes ghosts
        if owned_only:
            x = x[:n_owned]

        out = np.array(x, copy=copy)
        if reshape_blocks and bs > 1:
            out = out.reshape((-1, bs))
        return out

    # DOLFINx linear algebra vector (dolfinx.la.Vector)
    if hasattr(obj, "array") and hasattr(obj, "scatter_forward"):
        obj.scatter_forward()
        arr = obj.array
        return np.array(arr, copy=copy)

    # PETSc vector
    if PETSc is not None and isinstance(obj, PETSc.Vec):
        a = obj.getArray(readonly=True)
        return a.copy() if copy else a

    # pyadjoint scalar
    if isinstance(obj, pyadjoint.AdjFloat):
        return np.array(float(obj), dtype=np.float64)

    raise TypeError(f"Cannot convert type {type(obj)} to numpy.")


def numpy_to_fenics(a: np.ndarray, template, *, owned_only: bool = True):
    """
    Convert numpy -> DOLFINx object, using `template` as type/space guide.

    Notes:
      - For Functions, expects the array to match the OWNED dof count (by default).
      - For Constants, we update `template.value` (no need to recreate the Constant).
    """
    # DOLFINx Constant: update value (recommended style in DOLFINx)
    if isinstance(template, fem.Constant):
        # Ensure dtype compatibility
        template.value = np.asarray(a, dtype=np.asarray(template.value).dtype)
        return template

    # DOLFINx Function: create a new function on the same space and fill owned dofs
    if isinstance(template, fem.Function):
        V = template.function_space
        u = fem.Function(V)

        n_owned, _ = _owned_size(V)
        target_dtype = u.x.array.dtype

        flat = np.asarray(a, dtype=target_dtype).reshape(-1)
        if owned_only:
            if flat.size != n_owned:
                raise ValueError(
                    f"Wrong size for owned dofs: got {flat.size}, expected {n_owned}."
                )
            u.x.array[:n_owned] = flat
        else:
            # If you explicitly pass owned+ghost entries
            if flat.size != u.x.array.size:
                raise ValueError(
                    f"Wrong size (owned+ghost): got {flat.size}, expected {u.x.array.size}."
                )
            u.x.array[:] = flat

        # Populate ghosts from owners
        u.x.scatter_forward()
        return u

    # pyadjoint scalar
    if isinstance(template, pyadjoint.AdjFloat):
        return pyadjoint.AdjFloat(float(np.asarray(a)))

    raise TypeError(f"Cannot convert numpy array to {type(template)}.")
