#import fenics
import firedrake
from  firedrake import adjoint
#import fenics_adjoint
import numpy as np


def fenics_to_numpy(fenics_var):
    """Convert FEniCS variable to numpy array"""
    #if isinstance(fenics_var, (firedrake.Constant, adjoint.Constant)):
    if isinstance(fenics_var, (firedrake.Constant)):
        return fenics_var.values()

    #if isinstance(fenics_var, (firedrake.Function, adjoint.Constant)):
    if isinstance(fenics_var, (firedrake.Function,)):
        #np_array = fenics_var.vector().get_local()
        np_array = fenics_var.dat.data_ro
        assert isinstance(np_array,np.ndarray) # for multi space this is a tuple
        #n_sub = fenics_var.function_space().num_sub_spaces()
        n_sub = len(fenics_var.function_space().subspaces)
        # Reshape if function is multi-component
        if n_sub != 0:
            np_array = np.reshape(np_array, (len(np_array) // n_sub, n_sub))
        return np_array

    if isinstance(fenics_var, firedrake.GenericVector):
        return fenics_var.get_local()

    if isinstance(fenics_var, adjoint.AdjFloat):
        return np.array(float(fenics_var), dtype=np.float_)

    raise ValueError('Cannot convert ' + str(type(fenics_var)))


def numpy_to_fenics(numpy_array, fenics_var_template):
    """Convert numpy array to FEniCS variable"""
#    if isinstance(fenics_var_template, (firedrake.Constant, adjoint.Constant)):
    #if isinstance(fenics_var_template, (firedrake.Constant, adjoint.Constant)):
    if isinstance(fenics_var_template, (firedrake.Constant,)):
        if numpy_array.shape == (1,):
            return type(fenics_var_template)(numpy_array[0])
        else:
            return type(fenics_var_template)(numpy_array)

    if isinstance(fenics_var_template, (firedrake.Function, adjoint.Function)):
        np_n_sub = numpy_array.shape[-1]
        np_size = np.prod(numpy_array.shape)

        function_space = fenics_var_template.function_space()

        u = type(fenics_var_template)(function_space)
        fenics_size = u.vector().local_size()
        fenics_n_sub = function_space.num_sub_spaces()

        if (fenics_n_sub != 0 and np_n_sub != fenics_n_sub) or np_size != fenics_size:
            err_msg = 'Cannot convert numpy array to Function:' \
                      ' Wrong shape {} vs {}'.format(numpy_array.shape, u.vector().get_local().shape)
            raise ValueError(err_msg)

        if numpy_array.dtype != np.float_:
            err_msg = 'The numpy array must be of type {}, ' \
                      'but got {}'.format(np.float_, numpy_array.dtype)
            raise ValueError(err_msg)

        u.vector().set_local(np.reshape(numpy_array, fenics_size))
        u.vector().apply('insert')
        return u

    if isinstance(fenics_var_template, adjoint.AdjFloat):
        return adjoint.AdjFloat(numpy_array)

    err_msg = 'Cannot convert numpy array to {}'.format(fenics_var_template)
    raise ValueError(err_msg)
