#import fenics
#fenics.set_log_level(fenics.LogLevel.ERROR)
import firedrake
#firedrake.set_log_level(firedrake.LogLevel.ERROR)

from .torch_firedrake import FiredrakeModule

from .numpy_firedrake import firedrake_to_numpy
from .numpy_firedrake import numpy_to_firedrake
