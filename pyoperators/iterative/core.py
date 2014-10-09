from __future__ import absolute_import, division, print_function
"""
This module defines the base class IterativeAlgorithm.

"""

import collections
import numpy as np
import re

from ..utils import strenum, uninterruptible_if
from ..utils.mpi import MPI
from ..memory import empty
from .stopconditions import NoStopCondition

# Python 2 backward compatibility
try:
    range = xrange
except NameError:
    pass

__all__ = ['AbnormalStopIteration', 'IterativeAlgorithm']

NO_STOP_CONDITION = NoStopCondition()


class AbnormalStopIteration(Exception):
    pass


class IterativeAlgorithm(object):
    """
    Abstract class for iterative algorithms.

    In pseudo code, the flow of an IterativeAlgorithm is the following:

        niterations = order of recursion minus one
        try:
            initialize
            do infinite loop:
                test stop conditions
                niterations += 1
                try:
                    iteration defining the new state
                except StopIteration:
                    callback
                    update state
                    break
                except AbnormalStopIteration:
                    raise
                callback
                update state
        except StopIteration:
            pass
        except AbnormalStopIteration:
            raise
        return the output of the finalize method.

    The user can supply the following:
    - an initialize method
    - an iteration method
    - a normal stop condition
    - an abnormal stop condition
    - a callback function

    Attributes
    ----------
    info : dict
        This dictionary contains the recursion names of the recursion
        variables, their dtype and their shape.
    niterations : int
        Number of completed iterations. During the first pass, its value is
        equal to the recursion order minus one when the stop conditions are
        checked and to the recursion order in the iteration method and
        the callback function.
    order : int
        Recursion order. For example:
        order=1: x_new = f(x)
        order=2: x_new = f(x, x_old)
        order=3: x_new = f(x, x_old, x_old2)
        order=4: x_new = f(x, x_old, x_old2, x_old3)
        and so on.

    Methods
    -------
    initialize : Set variables to initial state.
    run : iterate until a stop condition is reached or Ctrl-C is pressed.
    cont : continue the algorithm
    restart : restart the algorithm if possible
    next : perform one (or more) iterations.
    callback : user-defined function to print status or save variables.

    """
    def __init__(self, allocate_new_state=True, callback=None,
                 clean_interrupt=True, disp=False, dtype=float,
                 inplace_recursion=False,
                 normal_stop_condition=NO_STOP_CONDITION,
                 abnormal_stop_condition=NO_STOP_CONDITION,
                 reuse_initial_state=False, **keywords):
        """
        Parameters
        ----------
        allocate_new_state : boolean, optional
            Tells if the buffers for the new state should be allocated
            beforehand. If true, the iteration method should reuse these
            buffers (ex: self.x_new[...] = ..) instead of creating new
            references (ex: self.x_new = ..).
        callback : callable, optional
            User-defined function to do actions such as printing status or
            plotting or saving variables. It is a callable with a single
            argument, namely the IterativeAlgorithm instance.
        clean_interrupt : boolean, optional
            An IterativeAlgorithm instance can be interrupted by pressing
            CTRL-C and still be restarted or iterated. There is a small
            overhead associated to it. To disable this feature, set this
            argument to False.
        disp : boolean
            If true, display iteration message
        dtype : numpy.dtype, optional
            Data type used to coerce the initial state variable to the same
            precision. It does not alter the data kind: complex variables stay
            complex.
        inplace_recursion : boolean, optional
            In some algorithm, it is not necessary to keep a copy of two
            states. It is then advisable to do the update in-place. For a given
            variable 'x', if the value of this argument is False, the variables
            'x_new' and 'x' will be available. If the value is True, only 'x'
            will be and the argument allocate_new_state has no effect.
        normal_stop_condition : StopCondition, optional
            The normal stop condition that will termintate the iteration.
        abnormal_stop_condition : StopCondition, optional
            The abnormal stop condition that will termintate the iteration.
            If such stop condition is met, an error message is printed and
        reuse_initial_state : boolean, optional
            Tells whether or not the buffers of the input initial state
            variables can be reused during these iterations. If True, beware
            of the side effects. Besides, the algorithm could not be restarted,
            as the initial state is lost.

        """
        self.clean_interrupt = clean_interrupt
        self.disp = False if MPI.COMM_WORLD.rank > 0 else disp
        self._set_buffer_handling(inplace_recursion, allocate_new_state,
                                  reuse_initial_state)
        self._set_order(keywords)
        self._set_variables(keywords)
        self._set_initial_state(keywords, dtype)
        self._set_callback(callback)
        self._set_stop_conditions(normal_stop_condition,
                                  abnormal_stop_condition)
        self.niterations = self.order - 1

    def __iter__(self):
        return self

    @staticmethod
    def callback(self):
        """ Callback function, called after each iteration.. """
        if not self.disp:
            return
        if self.inplace_recursion:
            current = self.finalize()
        elif len(self.variables) == 1:
            current = getattr(self, self.variables[0] + '_new')
        else:
            dict((v, getattr(self, v + '_new')) for v in self.variables)
        print('{0:4}: {1}'.format(self.niterations, current))

    def cont(self):
        """ Continue an interrupted computation. """
        if self.niterations == 0:
            raise RuntimeError("The iterative algorithm is not yet started. Us"
                               "e the 'run' method.")
        try:
            return self.next(np.iinfo(int).max)
        except StopIteration:
            pass
        except AbnormalStopIteration:
            raise
        return self.finalize()

    def finalize(self):
        """
        Perform some task at exit and return the value of the variables as
        a dictionary if there are more than one recursion variable and as
        the variable's value otherwise.

        """
        if len(self.variables) == 1:
            return getattr(self, self.variables[0])
        return dict((v, getattr(self, v)) for v in self.variables)

    def initialize(self):
        """
        Initialize the iterative algorithm by setting the initial values.

        """
        if self.niterations > self.order - 1 and self.reuse_initial_state:
            raise RuntimeError(
                'It is not possible to restart an algorithm for which the init'
                'ial state has not been saved. Instantiate the algorithme with'
                ' the keyword reuse_initial_state set to False.')
        self.niterations = self.order - 1
        self.success = True
        skip_new = not self.inplace_recursion

        # _set_buffer_handling scheme:
        # 1) copy=False, 2) True, 3) False, 4) False, 5) True
        copy = (self.inplace_recursion or self.allocate_new_state) and \
               not self.reuse_initial_state
        for var, info in self.info.items():
            for n, b in zip(info['names'][skip_new:],
                            self._initial_state[var]):
                #XXX FIXME: b should be aligned...
                b = np.array(b, info['dtype'], order='c', copy=copy)
                setattr(self, n, b)

    def next(self, n=1):
        """ Perform n iterations and return current solution. """
        if self.niterations == self.order - 1:
            self.initialize()
        for i in range(n):
            with uninterruptible_if(self.clean_interrupt):
                self._check_stop_conditions()
                self.niterations += 1
                try:
                    self.iteration()
                except StopIteration:
                    self.callback(self)
                    self._update_variables()
                    raise
                except AbnormalStopIteration:
                    raise
                self.callback(self)
                self._update_variables()
        return self.finalize()

    def __next__(self):
        return self.next()

    def iteration(self):
        """
        Algorithm actual iteration, It defines the new state from the previous
        ones.

        """
        raise NotImplementedError("The algorithm does not define an 'iteration"
                                  "' method.")

    def restart(self, n=None):
        """ Restart the algorithm. """
        self.initialize()
        return self.run(n)

    def run(self, n=None):
        """ Run the algorithm. """
        if self.niterations > self.order - 1:
            raise RuntimeError("The iterative algorithm is already started. Us"
                               "e the methods 'restart' or 'cont' instead.")
        n = n or np.iinfo(int).max
        try:
            return self.next(n)
        except StopIteration:
            pass
        except AbnormalStopIteration:
            raise
        return self.finalize()

    def _check_stop_conditions(self):
        """
        Raise a StopIteration if the normal stop condition is met. Raise an
        AbnormalStopIteration if the abnormal stop condition is met.

        """
        self.normal_stop_condition(self)
        try:
            self.abnormal_stop_condition(self)
        except StopIteration as e:
            raise AbnormalStopIteration(e)

    def _get_suffix(self):
        """
        Return list of string ['_new', '', '_old', '_old2, ...] according
        to recursion order.

        """
        if self.inplace_recursion:
            return ['']
        suffix = ['_new', '']
        if self.order == 1:
            return suffix
        suffix += ['_old']
        if self.order == 2:
            return suffix
        return suffix + ['_old{0}'.format(o-1) for o in range(3, self.order+1)]

    def _set_buffer_handling(self, inplace_recursion, allocate_new_state,
                             reuse_initial_state):
        """
        There are only 5 buffer handling schemes:
        1) out-of-place recursion, pre-allocate new state, the initial state
        buffers are reused during the iterations (IAR= False, True, True)
        2) out-of-place recursion, pre-allocate new state, the initial state
        is copied for the first iteration (IAR= False, True, False)
        3) out-of-place recursion, do not pre-allocate new state, the initial
        state is passed to the first iteration, where it should not be altered.
        (IAR= False, False, False)
        4) inplace recursion, reuse initial state (IAR= True, False, True)
        5) inplace recursion, do not reuse initial state (True, False, False)

        """
        self.allocate_new_state = allocate_new_state
        self.inplace_recursion = inplace_recursion
        self.reuse_initial_state = reuse_initial_state
        if inplace_recursion:
            self.allocate_new_state = False
        elif not allocate_new_state:
            self.reuse_initial_state = False

    def _set_callback(self, callback):
        """ Set the callback function, if specified. """
        if callback is None:
            return
        if not isinstance(callback, collections.Callable):
            raise TypeError('The callback function is not callable.')
        self.callback = callback

    def _set_initial_state(self, keywords, default_dtype):
        # _initial_state contains references to the input initial state:
        # no copy nor casting is done.
        self.info = {}
        self._initial_state = {}
        self._buffers = {} if self.allocate_new_state else None
        suffix = self._get_suffix()

        for var in self.variables:
            names = tuple(var + s for s in suffix)
            shapes = []
            initial_state = []
            dtype_fixed = keywords.get(var + '_dtype', None)
            dtype = np.dtype(dtype_fixed or default_dtype)
            skip_new = not self.inplace_recursion
            for n in names[skip_new:]:
                val = keywords[n]
                if not isinstance(val, np.ndarray):
                    val = np.array(val)
                # if the variable's dtype is not specified, we'll look for
                # promotion to complex from the initial values
                if dtype_fixed is None and dtype.kind == 'f' and \
                   val.dtype.kind == 'c':
                    dtype = np.dtype('complex' + str(2*int(dtype.name[5:])))
                shapes.append(val.shape)
                initial_state.append(keywords[n])

            shape = shapes[0]
            if any(s != shape for s in shapes[1:]):
                raise ValueError("The shapes of the initial values of '{0}' ar"
                                 "e incompatible: {1}.".format(var, shapes))

            self.info[var] = {'names': names,
                              'shape': shape,
                              'dtype': dtype}
            self._initial_state[var] = initial_state
            if self.allocate_new_state:
                setattr(self, var + '_new', empty(shape, dtype))

        # make sure that the initial buffers don't point the same memory loc.
        if self.reuse_initial_state:
            names = []
            addresses = []
            for var in self.variables:
                names += self.info[var]['names'][skip_new:]
                addresses += [b.__array_interface__['data'][0]
                              if isinstance(b, np.ndarray) else 0
                              for b in self._initial_state[var]]
            d = collections.defaultdict(list)
            for n, a in zip(names, addresses):
                d[a].append(n)
            duplicates = [v for k, v in d.items() if len(v) > 1 and k != 0]
            if len(duplicates) > 0:
                raise ValueError(
                    'Some initial values refer to the same buffer: {0}.'.
                    format(strenum(('='.join(d) for d in duplicates), 'and')))

    def _set_order(self, keywords):
        """ Set the order of the recursion. """
        order = 1
        if any(k.endswith('_old') for k in keywords):
            order += 1
            while any(k.endswith('_old'+str(order)) for k in keywords):
                order += 1
        self.order = order

    def _set_stop_conditions(self, normal_stop, abnormal_stop):
        """ Set the stop conditions. """
        if not isinstance(normal_stop, collections.Callable) or not isinstance(abnormal_stop, collections.Callable):
            raise TypeError('The stop conditions must be callable.')
        self.normal_stop_condition = normal_stop
        self.abnormal_stop_condition = abnormal_stop

    def _set_variables(self, keywords):
        """ Set the variable names of the recursion. """
        regex = re.compile(r'^((?!(_old[0-9]*|_new|_dtype)$).)*$')
        variables = list(set(k for k in keywords if regex.match(k)))
        variables.sort()

        suffix = self._get_suffix()
        for var in variables:
            for s in suffix:
                if s != '_new' and var + s not in keywords:
                    raise ValueError("The initial value '{0}' is not specified"
                                     ".".format(var + s))
        self.variables = variables

    def _update_variables(self):
        """ Cyclic update of the variables. """
        if self.inplace_recursion:
            return
        for var in self.variables:
            names = self.info[var]['names']
            buffers = tuple(getattr(self, n) for n in names)
            setattr(self, names[0], buffers[-1])
            for n, b in zip(names[1:], buffers[:-1]):
                setattr(self, n, b)
