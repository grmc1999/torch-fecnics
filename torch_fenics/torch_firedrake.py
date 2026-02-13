from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch

import firedrake as fd
from firedrake import adjoint
from .numpy_firedrake import numpy_to_firedrake, firedrake_to_numpy
#import numpy_to_firedrake

class FiredrakeFunction(torch.autograd.Function):
    """Executes the solve() method of a FiredrakeModule and differentiates with firedrake.adjoint."""

    @staticmethod
    def forward(ctx, firedrake_solver: "FiredrakeModule", *args):
        # 1) Check arity
        n_args = len(args)
        expected = len(firedrake_solver.firedrake_input_templates())
        if n_args != expected:
            raise ValueError(
                f"Wrong number of arguments to {firedrake_solver}. Expected {expected} got {n_args}."
            )

        # 2) Check shapes against numpy templates
        numpy_templates = firedrake_solver.numpy_input_templates()
        for i, (arg, templ) in enumerate(zip(args, numpy_templates)):
            if tuple(arg.shape) != tuple(templ.shape):
                raise ValueError(
                    f"Expected input shape {templ.shape} for input {i} but got {tuple(arg.shape)}."
                )

        # 3) Enforce float64 (common Firedrake default)
        for i, arg in enumerate(args):
            if (isinstance(arg, np.ndarray) and arg.dtype != np.float64) or (
                torch.is_tensor(arg) and arg.dtype != torch.float64
            ):
                raise TypeError(
                    f"All inputs must be torch.float64 / np.float64, but got {arg.dtype} for input {i}."
                )

        # 4) Convert torch -> numpy -> Firedrake
        firedrake_inputs = []
        for inp, templ in zip(args, firedrake_solver.firedrake_input_templates()):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            firedrake_inputs.append(numpy_to_firedrake(inp, templ))

        # 5) Start a fresh tape for this forward pass (robust across repeated calls)
        # Prefer new Tape if available; otherwise clear working tape.
        if hasattr(adjoint, "Tape") and hasattr(adjoint, "set_working_tape"):
            tape = adjoint.Tape()
            adjoint.set_working_tape(tape)
        else:
            tape = adjoint.get_working_tape()
            tape.clear_tape()

        adjoint.continue_annotation()

        # 6) Run Firedrake solve
        firedrake_outputs = firedrake_solver.solve(*firedrake_inputs)

        adjoint.pause_annotation()

        # Normalize outputs to tuple
        if not isinstance(firedrake_outputs, tuple):
            firedrake_outputs = (firedrake_outputs,)

        # Save for backward
        ctx.tape = tape
        ctx.firedrake_inputs = firedrake_inputs
        ctx.firedrake_outputs = firedrake_outputs

        # 7) Return torch tensors
        outs = tuple(
            torch.from_numpy(firedrake_to_numpy(out)).to(dtype=torch.float64)
            for out in firedrake_outputs
        )
        return outs[0] if len(outs) == 1 else outs

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Compute gradients w.r.t. Firedrake inputs via VJP:
          dL/dm = sum_k <grad_output_k, d output_k / d m>
        implemented as compute_gradient(output_k, controls, adj_value=seed_k).
        """
        # Map which inputs need grads (skip index 0: solver object)
        needs = ctx.needs_input_grad  # tuple aligned with forward inputs (solver, *args)
        arg_needs = needs[1:]

        # Build (input_index, Control) list
        ctrl_items: List[Tuple[int, Any]] = []
        for i, (need_g, inp) in enumerate(zip(arg_needs, ctx.firedrake_inputs), start=1):
            if need_g:
                ctrl_items.append((i, adjoint.Control(inp)))

        if not ctrl_items:
            # No gradients requested
            return (None,) + tuple(None for _ in ctx.firedrake_inputs)

        ctrl_indices = [i for i, _ in ctrl_items]
        controls = [c for _, c in ctrl_items]

        # Prepare accumulation buffers (torch tensors) per requested control
        acc: List[Union[None, torch.Tensor]] = [None] * len(controls)

        # For each output, apply its adjoint seed (grad_output) and accumulate grads
        for grad_out, out in zip(grad_outputs, ctx.firedrake_outputs):
            if grad_out is None:
                continue

            seed_np = grad_out.detach().cpu().numpy()
            adj_value = numpy_to_firedrake(seed_np, out)

            # Compute gradient of "out" w.r.t. controls, applying adjoint seed adj_value
            # (Vector–Jacobian product.)
            grads = adjoint.compute_gradient(out, controls, tape=ctx.tape, adj_value=adj_value)

            # Convert and accumulate
            for j, g in enumerate(grads):
                if g is None:
                    continue
                tg = torch.from_numpy(firedrake_to_numpy(g)).to(dtype=torch.float64)
                acc[j] = tg if acc[j] is None else (acc[j] + tg)

        # Build return tuple aligned with forward inputs: (None for solver, grad for each arg)
        result: List[Union[None, torch.Tensor]] = [None] * (1 + len(ctx.firedrake_inputs))
        # Fill computed grads at their corresponding arg positions
        for (pos, _), g in zip(ctrl_items, acc):
            result[pos] = g

        return tuple(result)


class FiredrakeModule(ABC, torch.nn.Module):
    """
    Base class: implement input_templates() and solve().

    - input_templates(): return a Firedrake object (or tuple) that defines input types/spaces.
    - solve(*args): run the Firedrake solve and return output(s) (Function/Constant/AdjFloat/etc.)
    """

    def __init__(self):
        super().__init__()
        self._fd_input_templates: Union[None, Tuple[Any, ...]] = None
        self._np_input_templates: Union[None, List[np.ndarray]] = None

    @abstractmethod
    def input_templates(self) -> Union[Any, Tuple[Any, ...]]:
        pass

    @abstractmethod
    def solve(self, *args) -> Union[Any, Tuple[Any, ...]]:
        pass

    def firedrake_input_templates(self) -> Tuple[Any, ...]:
        if self._fd_input_templates is None:
            temps = self.input_templates()
            if not isinstance(temps, tuple):
                temps = (temps,)
            self._fd_input_templates = temps
        return self._fd_input_templates

    def numpy_input_templates(self) -> List[np.ndarray]:
        if self._np_input_templates is None:
            self._np_input_templates = [firedrake_to_numpy(t) for t in self.firedrake_input_templates()]
        return self._np_input_templates

    def forward(self, *args: torch.Tensor):
        """
        Batched execution:
          each arg is [N, ...template_shape...]
        """
        if len(args) != 0:
            n = args[0].shape[0]
            for arg in args[1:]:
                if arg.shape[0] != n:
                    raise ValueError("Batch size must be the same for each input argument.")

        # Run per-sample (simple + safe with tapes; you can optimize later)
        outs = [FiredrakeFunction.apply(self, *inp) for inp in zip(*args)]

        # Normalize to tuple-of-outputs and stack across batch
        if len(outs) == 0:
            raise ValueError("No inputs provided.")
        if torch.is_tensor(outs[0]):
            return torch.stack(outs)
        else:
            # multiple outputs
            return tuple(torch.stack(o) for o in zip(*outs))
