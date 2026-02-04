#import fenics
#fenics.set_log_level(fenics.LogLevel.ERROR)
import firedrake
firedrake.set_log_level(firedrake.LogLevel.ERROR)

from .torch_fenics import FEniCSModule

from .numpy_fenics import fenics_to_numpy
from .numpy_fenics import numpy_to_fenics
