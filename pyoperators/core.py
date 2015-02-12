#coding: utf-8
"""
The core module defines the Operator class. Operators are functions
which can be added, composed or multiplied by a scalar. See the
Operator docstring for more information.
"""

from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
import operator
import pyoperators as po
import scipy.sparse as sp
import sys
import types
from collections import MutableMapping, MutableSequence, MutableSet
from itertools import groupby
from . import config
from .flags import (
    Flags, idempotent, inplace, involutary, linear, real,
    square, symmetric, update_output)
from .memory import (
    empty, garbage_collect, iscompatible, zeros, MemoryPool)
from .utils import (
    all_eq, first_is_not, inspect_special_values, isalias, isclassattr,
    isscalarlike, merge_none, ndarraywrap, operation_assignment, product,
    renumerate, strenum, strplural, strshape, Timer, tointtuple)
from .utils.mpi import MPI
import collections

__all__ = [
    'Operator',
    'AdditionOperator',
    'BlockColumnOperator',
    'BlockDiagonalOperator',
    'BlockRowOperator',
    'BlockSliceOperator',
    'CompositionOperator',
    'ConstantOperator',
    'DiagonalOperator',
    'GroupOperator',
    'HomothetyOperator',
    'IdentityOperator',
    'MultiplicationOperator',
    'ReshapeOperator',
    'ReductionOperator',
    'Variable',
    'ZeroOperator',
    'asoperator',
    'timer_operator',
]

DEBUG = 0

OPERATOR_ATTRIBUTES = ['attrin', 'attrout', 'classin', 'classout', 'commin',
                       'commout', 'reshapein', 'reshapeout', 'shapein',
                       'shapeout', 'toshapein', 'toshapeout', 'validatein',
                       'validateout', 'dtype', 'flags']


class Operator(object):
    """
    Operator top-level class.

    The operator class is a function factory.

    Attributes
    ----------
    attrin/attrout : dict or function
        If attrout is a dict, its items are added to the output. If it is
        a function, it takes the input attributes and returns the output attri-
        butes. The attrin attribute is only used in the reversed direction.
    classin/classout : ndarray subclass
        The classout attribute sets the output class. The classin attribute is
        only used in the reversed direction.
    commin/commout : mpi4py.Comm
        The commin and commout attributes store the MPI communicator for the
        input and output.
    reshapein/reshapeout : function
        The reshapein function takes the input shape and returns the output
        shape. The method is used for implicit output shape operators.
        The reshapeout function does the opposite.
    shapein : tuple
        Operator's input shape.
    shapeout : tuple
        Operator's output shape.
    toshapein/toshapeout : function
        The toshapein function reshapes a vector into a multi-dimensional array
        compatible with the operator's input shape. The toshapeout method is
        only used in the reversed direction.
    validatein/validateout : function
        The validatein function raises a ValueError exception if the input
        shape is not valid. The validateout function is used in the reversed
        direction
    flags : Flags
        The flags describe properties of the operator.
    dtype : dtype
        The operator's dtype is used to determine the dtype of its output.
        Unless it is None, the output dtype is the common type of the operator
        and input dtypes. If dtype is None, the output dtype is the input
        dtype.
    C : Operator
        Oonjugate operator.
    T : Operator
        Tranpose operator.
    H : Operator
        Adjoint operator.
    I : Operator
        Inverse operator.

    """
    def __init__(self, direct=None, transpose=None, adjoint=None,
                 conjugate=None, inverse=None, inverse_transpose=None,
                 inverse_adjoint=None, inverse_conjugate=None,
                 attrin={}, attrout={}, classin=None, classout=None,
                 commin=None, commout=None, reshapein=None, reshapeout=None,
                 shapein=None, shapeout=None, toshapein=None, toshapeout=None,
                 validatein=None, validateout=None, dtype=None, flags={},
                 name=None):
        for method, name_ in zip(
            (direct, transpose, adjoint, conjugate, inverse,
             inverse_transpose, inverse_adjoint, inverse_conjugate),
            ('direct', 'transpose', 'adjoint', 'conjugate', 'inverse',
             'inverse_transpose', 'inverse_adjoint', 'inverse_conjugate')):
            if method is not None:
                if not hasattr(method, '__call__'):
                    raise TypeError("The method '%s' is not callable." % name_)
                # should also check that the method has at least two arguments
                setattr(self, name_, method)

        self._init_dtype(dtype)
        self._init_flags(flags)
        self._init_rules()
        self._init_name(name)
        self._init_inout(attrin, attrout, classin, classout, commin, commout,
                         reshapein, reshapeout, shapein, shapeout, toshapein,
                         toshapeout, validatein, validateout)

    __name__ = None
    dtype = None
    flags = Flags()
    rules = None

    _C = None
    _T = None
    _H = None
    _I = None

    attrin = {}
    attrout = {}
    classin = None
    classout = None
    commin = None
    commout = None
    shapein = None
    shapeout = None

    def delete(self):
        """
        Delete an operator and its associated operators.

        The operators are morphed into empty shell DeletedOperators and
        a garbage collection may be triggered according the operator
        memory footprints.

        """
        if self._C is None:
            operators = (self,)
        else:
            operators = (self, self._C, self._T, self._H, self._I, self._I._C,
                         self._I._T, self._I._H)
        for operator in operators:
            nbytes = operator.nbytes
            operator.__class__ = DeletedOperator
            del operator.__dict__
            garbage_collect(nbytes)

    @property
    def nbytes(self):
        """
        Approximate memory footprint.

        """
        return 0

    def reshapein(self, shape):
        """
        Return the output shape given an input shape.

        Parameter
        ---------
        shape : tuple
           The input shape. It is guaranteed 1) not to be None although this
           method returns None if and only if the operator's output shape
           is unconstrained and 2) to be a tuple.

        Note
        ----
        Implicit output shape operators do override this method.

        """
        return self.shapeout

    def reshapeout(self, shape):
        """
        Return the input shape given an output shape.

        Parameter
        ---------
        shape : tuple
           The output shape. It is guaranteed 1) not to be None although this
           method returns None if and only if the operator's input shape
           is unconstrained and 2) to be a tuple.

        Note
        ----
        Implicit input shape operators do override this method.

        """
        return self.shapein

    def toshapein(self, v):
        """
        Reshape a vector into a multi-dimensional array compatible with
        the operator's input shape.

        """
        if self.shapein is None:
            raise ValueError("The operator '" + self.__name__ + "' does not ha"
                             "ve an explicit shape.")
        return v.reshape(self.shapein)

    def toshapeout(self, v):
        """
        Reshape a vector into a multi-dimensional array compatible with
        the operator's output shape.

        """
        if self.shapeout is None:
            raise ValueError("The operator '" + self.__name__ + "' does not ha"
                             "ve an explicit shape.")
        return v.reshape(self.shapeout)

    def propagate_attributes(self, cls, attr):
        """
        Propagate attributes according to operator's attrout. If the class
        changes, class attributes are removed if they are not class attributes
        of the new class.
        """
        if None not in (self.classout, cls) and self.classout is not cls:
            for a in list(attr.keys()):
                if isclassattr(a, cls) and not isclassattr(a, self.classout):
                    del attr[a]
        if 'shape_global' in attr:
            del attr['shape_global']
        if isinstance(self.attrout, dict):
            for k, v in self.attrout.items():
                if isinstance(v, (MutableMapping, MutableSequence,
                                  MutableSet)):
                    if hasattr(v, 'copy'):
                        v = v.copy()
                    elif type(v) is list:
                        v = list(v)
                attr[k] = v
        else:
            self.attrout(attr)
        return self.classout or cls

    def propagate_commin(self, commin):
        """
        Propagate MPI communicator of the input to the operands.
        Operands have the possibility to change during this step.

        """
        return self

    def propagate_commout(self, commin):
        """
        Propagate MPI communicator of the output to the operands.
        Operands have the possibility to change during this step.

        """
        return self

    def validatein(self, shapein):
        """
        Validate an input shape by raising a ValueError exception if it is
        invalid.

        """
        if self.shapein is not None and self.shapein != shapein:
            raise ValueError(
                "The input shape '{0}' is incompatible with that of {1}: '{2}'"
                ".".format(shapein, self.__name__, self.shapein))

    def validateout(self, shapeout):
        """
        Validate an output shape by raising a ValueError exception if it is
        invalid.

        """
        if self.shapeout is not None and self.shapeout != shapeout:
            raise ValueError(
                "The output shape '{0}' is incompatible with that of {1}: '{2"
                "}'.".format(shapeout, self.__name__, self.shapeout))

    # for the next methods, the following always stand:
    #    - input and output are not in the memory pool
    #    - input and output are compatible with the operator's requirements
    #      in terms of shape, contiguity and alignment.
    direct = None

    def conjugate(self, input, output):
        if input.dtype.kind == 'c':
            with _pool.get(input.shape, input.dtype) as buf:
                np.conjugate(input, buf)
            input = buf
        self.direct(input, output)
        np.conjugate(output, output)

    transpose = None
    adjoint = None
    inverse = None
    inverse_conjugate = None
    inverse_transpose = None
    inverse_adjoint = None

    def __call__(self, x, out=None, operation=operation_assignment,
                 preserve_input=True):

        if isinstance(x, Operator):
            if self.flags.idempotent and self is x:
                return self
            return CompositionOperator([self, x])

        if self.direct is None:
            raise NotImplementedError('Call to ' + self.__name__ + ' is not im'
                                      'plemented.')

        if operation is not operation_assignment:
            if not self.flags.update_output:
                raise ValueError(
                    'This operator does not handle inplace reductions.')
            if out is None:
                raise ValueError(
                    'The output placeholder is not specified.')

        with timer_operator:
            # get valid input and output
            i, i_, o, o_ = self._validate_arguments(x, out)

            # perform computation
            reuse_x = isinstance(x, np.ndarray) and not isalias(x, i) and \
                not preserve_input
            reuse_out = isinstance(out, np.ndarray) and not isalias(out, i) \
                and not isalias(out, o)

            with _pool.set_if(reuse_x, x):
                with _pool.set_if(reuse_out, out):
                    if self.flags.update_output:
                        self.direct(i, o, operation=operation)
                    else:
                        self.direct(i, o)

            # add back temporaries for input & output in the memory pool
            if i_ is not None:
                _pool.add(i_)
            if out is None:
                out = o
            elif not isalias(out, o):
                out[...] = o
                _pool.add(o_)

            # copy over class and attributes
            cls = x.__class__ if isinstance(x, np.ndarray) else np.ndarray
            attr = x.__dict__.copy() if hasattr(x, '__dict__') else {}
            cls = self.propagate_attributes(cls, attr)
            if cls is np.ndarray and len(attr) > 0:
                cls = ndarraywrap
            if out is None:
                out = o
            if type(out) is np.ndarray:
                if cls is np.ndarray:
                    return out
                out = out.view(cls)
            elif type(out) is not cls:
                out.__class__ = cls
                if out.__array_finalize__ is not None:
                    out.__array_finalize__()

            # we cannot simply update __dict__, because of properties.
            # the iteration is sorted by key, so that attributes beginning with
            # an underscore are set first.
            for k in sorted(attr.keys()):
                setattr(out, k, attr[k])
            return out

    @property
    def shape(self):
        return (product(self.shapeout), product(self.shapein))

    def todense(self, shapein=None, shapeout=None, inplace=False):
        """
        Output the dense representation of the Operator as a ndarray.

        Arguments
        ---------
        shapein : tuple of ints, (default: None)
            The operator's input shape if it is not explicit.
        shapeout : tuple of ints (default: None)
            The operator's output shape if it is not explicit.
        inplace : boolean
            For testing purposes only. By default, this method uses
            out-of-place operations that directly fill the output array.
            By setting inplace to True, one can test in-place operations, at
            the cost of additional copies.

        """
        shapein, shapeout = self._validate_shapes(shapein, shapeout)
        if shapein is None:
            raise ValueError("The operator's input shape is not explicit. Spec"
                             "ify it with the 'shapein' keyword.")
        if shapeout is None:
            raise ValueError("The operator's output shape is not explicit. Spe"
                             "cify it with the 'shapeout' keyword.")
        m, n = product(shapeout), product(shapein)
        dtype = int if self.dtype is None else self.dtype
        d = np.empty((n, m), dtype)

        if not inplace or not self.flags.inplace:
            v = zeros(n, dtype)
            if not self.flags.aligned_output:
                for i in range(n):
                    v[i] = 1
                    o = d[i, :].reshape(shapeout)
                    self.direct(v.reshape(shapein), o)
                    v[i] = 0
            else:
                o = empty(shapeout, dtype)
                for i in range(n):
                    v[i] = 1
                    self.direct(v.reshape(shapein), o)
                    d[i, :] = o.ravel()
                    v[i] = 0
            return d.T

        # test in-place mechanism
        u = empty(max(m, n), dtype)
        v = u[:n]
        w = u[:m]
        for i in range(n):
            v[:] = 0
            v[i] = 1
            self.direct(v.reshape(shapein), w.reshape(shapeout))
            d[i, :] = w
        return d.T

    def matvec(self, x, out=None):
        assert not isinstance(x, np.ndarray) or x.flags.contiguous
        assert out is None or \
            isinstance(out, np.ndarray) and out.flags.contiguous
        x = self.toshapein(x)
        if out is not None:
            out = self.toshapeout(out)
        out = self.__call__(x, out=out)
        return out.ravel()

    def rmatvec(self, x, out=None):
        return self.T.matvec(x, out=out)

    def set_rule(self, subjects, predicate, operation=None):
        """
        Add a rule to the rule list, taking care of duplicates and priorities.
        Class-matching rules have a lower priority than the others.

        Parameters
        ----------
        subjects : str
            See UnaryRule and BinaryRule documentation.
        predicate : str
            See UnaryRule and BinaryRule documentation.
        operation : CompositeOperator sub class
            Operation to which applies the rule. It can be:
                - None, for unary rules
                - CompositionOperator
                - AdditionOperator
                - MultiplicationOperator.

        """
        # Handle first the case of multiple subclass matching rules
        if isinstance(subjects, (list, tuple)) and len(subjects) == 2:
            if isinstance(subjects[0], (list, tuple)):
                for s in subjects[0][::-1]:
                    self.set_rule((s, subjects[1]), predicate,
                                  operation=operation)
                return
            if isinstance(subjects[1], (list, tuple)):
                for s in subjects[1][::-1]:
                    self.set_rule((subjects[0], s), predicate,
                                  operation=operation)
                return

        rule = po.rules.Rule(subjects, predicate)

        if len(rule.subjects) > 2:
            raise ValueError('Only unary and binary rules are allowed.')

        if operation is None and len(rule.subjects) == 2:
            raise ValueError('The operation is not specified.')

        # get the rule list for the specified operation
        if operation is None:
            if None not in self.rules:
                self.rules[None] = []
            rules = self.rules[None]
        elif issubclass(operation, CommutativeCompositeOperator):
            if rule.subjects[-1] == '.':
                rule.subjects = rule.subjects[::-1]
                rule.reference = 0
            if operation not in self.rules:
                self.rules[operation] = []
            rules = self.rules[operation]
        else:
            if operation not in self.rules:
                self.rules[operation] = {'left': [], 'right': []}
            rules = self.rules[operation]['left' if rule.reference == 0 else
                                          'right']
        ids = [r.subjects for r in rules]

        # first, try to override existing rule
        try:
            index = ids.index(rule.subjects)
            rules[index] = rule
            return
        except ValueError:
            pass

        # class matching rules have lower priority
        if len(rule.subjects) == 1 or \
           isinstance(rule.other, str) and not rule.other.startswith('{'):
            rules.insert(0, rule)
            return

        # search for subclass rules
        for index, r in enumerate(rules):
            if isinstance(r.other, type):
                break
        else:
            rules.append(rule)
            return

        # insert the rule after more specific ones
        cls = rule.other
        classes = [r.other for r in rules[index:]]
        is_subclass = [issubclass(cls, c) for c in classes]
        is_supclass = [issubclass(c, cls) for c in classes]
        try:
            index2 = is_subclass.index(True)
        except ValueError:
            try:
                index2 = len(is_supclass) - is_supclass[::-1].index(True)
            except ValueError:
                index2 = 0
        rules.insert(index + index2, rule)

    def del_rule(self, subjects, operation=None):
        """
        Delete an operator rule.

        If the rule does not exist, a ValueError exception is raised.

        Parameters
        ----------
        subjects : str
            The subjects of the rule to be deleted.
        operation : CompositeOperator sub class
            Operation to which applies the rule to be deleted. It can be:
            CompositionOperator, AdditionOperator and MultiplicationOperator.
            For unary rules, the value must be None.
        """
        subjects = po.rules.Rule._split_subject(subjects)
        if len(subjects) > 2:
            raise ValueError('Only unary and binary rules are allowed.')
        if operation is None and len(subjects) == 2:
            raise ValueError('The operation is not specified.')
        if operation not in self.rules:
            if None not in self.rules:
                raise ValueError('There is no unary rule.')
            raise ValueError("The operation '{0}' has no rules.".format(type(
                             operation).__name__))
        rules = self.rules[operation]
        if operation is not None:
            right = subjects[-1] == '.'
            if issubclass(operation, CommutativeCompositeOperator):
                if right:
                    subjects = subjects[::-1]
            else:
                rules = rules['right' if right else 'left']
        index = [r.subjects for r in rules].index(subjects)
        del rules[index]

    @property
    def C(self):
        """ Return the complex-conjugate of the operator. """
        if self._C is None:
            self._generate_associated_operators()
        return self._C

    @property
    def T(self):
        """ Return the transpose of the operator. """
        if self._T is None:
            self._generate_associated_operators()
        return self._T

    @property
    def H(self):
        """ Return the adjoint of the operator. """
        if self._H is None:
            self._generate_associated_operators()
        return self._H

    @property
    def I(self):
        """ Return the inverse of the operator. """
        if self._I is None:
            self._generate_associated_operators()
        return self._I

    def copy(self, target=None):
        """ Return a shallow copy of the operator. """
        class Target(object):
            pass
        if target is None:
            target = Target()
        target.__class__ = self.__class__
        for k, v in self.__dict__.items():
            if k in ('_C', '_T', '_H', '_I'):
                continue
            if isinstance(v, types.MethodType) and v.__self__ is self:
                target.__dict__[k] = types.MethodType(v.__func__, target)
            else:
                target.__dict__[k] = v
        return target

    @staticmethod
    def _find_common_type(dtypes):
        """ Return dtype of greater type rank. """
        dtypes = [d for d in dtypes if d is not None]
        if len(dtypes) == 0:
            return None
        return np.find_common_type(dtypes, [])

    def _generate_associated_operators(self):
        """
        Compute at once the conjugate, transpose, adjoint and inverse operators
        of the instance and of themselves.

        """
        rules = dict((r.subjects[0], r) for r in self.rules.get(None, {}))
        flags = self.flags

        if flags.real:
            C = self
        elif 'C' in rules:
            C = _copy_direct(self, rules['C'](self))
        else:
            C = _copy_direct_all(
                self, Operator(direct=self.conjugate,
                               name=self.__name__ + '.C',
                               flags={'linear': flags.linear,
                                      'symmetric': flags.symmetric,
                                      'hermitian': flags.hermitian,
                                      'idempotent': flags.idempotent,
                                      'involutary': flags.involutary,
                                      'orthogonal': flags.orthogonal,
                                      'unitary': flags.unitary}))

        new_flags = {
            'linear': flags.linear,
            'idempotent': flags.idempotent,
            'involutary': flags.involutary,
            'orthogonal': flags.orthogonal,
            'unitary': flags.unitary}
        if flags.symmetric:
            T = self
        elif 'T' in rules:
            T = _copy_reverse(self, rules['T'](self))
        elif flags.real and 'H' in rules:
            T = _copy_reverse(self, rules['H'](self))
        elif flags.orthogonal and 'I' in rules:
            T = _copy_reverse(self, rules['I'](self))
        elif self.transpose is not None:
            T = _copy_reverse_all(
                self, Operator(direct=self.transpose,
                               name=self.__name__ + '.T', flags=new_flags))
        else:
            T = None

        if flags.hermitian:
            H = self
        elif flags.symmetric:
            H = C
        elif flags.real:
            H = T
        elif 'H' in rules:
            H = _copy_reverse(self, rules['H'](self))
        elif flags.unitary and 'I' in rules:
            H = _copy_reverse(self, rules['I'](self))
        elif self.adjoint is not None:
            H = _copy_reverse_all(
                self, Operator(direct=self.adjoint,
                               name=self.__name__ + '.H', flags=new_flags))
        else:
            H = None

        if T is None:
            if H is not None:
                if flags.real:
                    T = H
                else:
                    T = _copy_reverse_all(
                        self, Operator(direct=H.conjugate, name=
                                       self.__name__ + '.T', flags=new_flags))
            else:
                T = _copy_reverse_all(
                    self, Operator(name=self.__name__ + '.T', flags=new_flags))
                if flags.real:
                    H = T

        if H is None:
            H = _copy_reverse_all(
                self, Operator(direct=T.conjugate if T is not None else None,
                               name=self.__name__ + '.H', flags=new_flags))

        if flags.involutary:
            I = self
        elif flags.orthogonal:
            I = T
        elif flags.unitary:
            I = H
        elif 'I' in rules:
            I = _copy_reverse(self, rules['I'](self))
        else:
            I = _copy_reverse_all(
                self, Operator(direct=self.inverse,
                               name=self.__name__ + '.I',
                               flags={'linear': flags.linear,
                                      'idempotent': flags.idempotent,
                                      'involutary': flags.involutary,
                                      'orthogonal': flags.orthogonal,
                                      'unitary': flags.unitary}))

        new_flags = {
            'idempotent': flags.idempotent,
            'involutary': flags.involutary,
            'orthogonal': flags.orthogonal,
            'unitary': flags.unitary}
        if flags.real:
            IC = I
        elif flags.orthogonal:
            IC = H
        elif flags.unitary:
            IC = T
        elif flags.involutary:
            IC = C
        elif 'IC' in rules:
            IC = _copy_reverse(self, rules['IC'](self))
        else:
            if self.inverse_conjugate is not None:
                func = self.inverse_conjugate
            elif I is not None:
                func = I.conjugate
            else:
                func = None
            IC = _copy_reverse_all(
                self, Operator(direct=func, name=self.__name__ + '.I.C',
                               flags=new_flags))

        if flags.orthogonal:
            IT = self
        elif flags.symmetric:
            IT = I
        elif flags.unitary:
            IT = C
        elif flags.involutary:
            IT = T
        elif 'IT' in rules:
            IT = _copy_direct(self, rules['IT'](self))
        elif self.inverse_transpose is not None:
            IT = _copy_direct_all(
                self, Operator(direct=self.inverse_transpose,
                               name=self.__name__ + '.I.T', flags=new_flags))
        else:
            IT = None

        if flags.unitary:
            IH = self
        elif flags.hermitian:
            IH = I
        elif flags.orthogonal:
            IH = C
        elif flags.involutary:
            IH = H
        elif flags.symmetric:
            IH = IC
        elif flags.real:
            IH = IT
        elif 'IH' in rules:
            IH = _copy_direct(self, rules['IH'](self))
        elif self.inverse_adjoint is not None:
            IH = _copy_direct_all(
                self, Operator(direct=self.inverse_adjoint,
                               name=self.__name__ + '.I.H', flags=new_flags))
        else:
            IH = None

        if IT is None:
            if IH is not None:
                if flags.real:
                    IT = IH
                else:
                    IT = _copy_direct_all(
                        self, Operator(direct=IH.conjugate,
                                       name=self.__name__ + '.I.T',
                                       flags=new_flags))
            else:
                IT = _copy_direct_all(
                    self, Operator(name=self.__name__ + '.I.T',
                                   flags=new_flags))
                if flags.real:
                    IH = IT

        if IH is None:
            IH = _copy_direct_all(
                self, Operator(direct=IT.conjugate if IT is not None else None,
                               name=self.__name__ + '.I.H', flags=new_flags))

        # once all the associated operators are instanciated, we set all their
        # associated operators. To do so, we use the fact that the transpose,
        # adjoint, conjugate and inverse operators are commutative and
        # involutary.
        self._C, self._T, self._H, self._I = C, T, H, I
        C._C, C._T, C._H, C._I = self, H, T, IC
        T._C, T._T, T._H, T._I = H, self, C, IT
        H._C, H._T, H._H, H._I = T, C, self, IH
        I._C, I._T, I._H, I._I = IC, IT, IH, self
        IC._C, IC._T, IC._H, IC._I = I, IH, IT, C
        IT._C, IT._T, IT._H, IT._I = IH, I, IC, T
        IH._C, IH._T, IH._H, IH._I = IT, IC, I, H

    def _init_dtype(self, dtype):
        if dtype is not None:
            dtype = np.dtype(dtype)
        self.dtype = dtype

    def _init_flags(self, flags):

        self._set_flags(flags)

        # A non-complex dtype sets the real flag to true.
        if self.dtype is None or self.dtype.kind != 'c':
            self._set_flags('real')

        if self.flags.real:
            if self.flags.symmetric:
                self._set_flags('hermitian')
            if self.flags.hermitian:
                self._set_flags('symmetric')
            if self.flags.orthogonal:
                self._set_flags('unitary')
            if self.flags.unitary:
                self._set_flags('orthogonal')

        if self.flags.orthogonal:
            if self.flags.idempotent:
                self._set_flags('symmetric')
            if self.flags.symmetric:
                self._set_flags('idempotent')

        if self.flags.unitary:
            if self.flags.idempotent:
                self._set_flags('hermitian')
            if self.flags.hermitian:
                self._set_flags('idempotent')

        if self.flags.involutary:
            if self.flags.symmetric:
                self._set_flags('orthogonal')
            if self.flags.orthogonal:
                self._set_flags('symmetric')
            if self.flags.hermitian:
                self._set_flags('unitary')
            if self.flags.unitary:
                self._set_flags('hermitian')

        if self.flags.idempotent:
            if any([self.flags.orthogonal, self.flags.unitary,
                    self.flags.involutary]):
                self._set_flags('orthogonal, unitary, involutary')

        if isinstance(flags, (dict, str)):
            auto_flags = ('shape_input', 'shape_output')
            mask = [f in flags for f in auto_flags]
            if any(mask):
                raise ValueError(
                    'The {0} {1} cannot be set.'.format(
                        strplural(np.sum(mask), 'flag', nonumber=True),
                        strenum([a for a, m in zip(auto_flags, mask) if m])))

        if isinstance(self.direct, np.ufunc):
            if self.direct.nin != 1 or self.direct.nout != 1:
                raise TypeError('A ufunc with several inputs or outputs cannot'
                                ' be converted to an Operator.')
            real = True
            if all(_[3] in 'EFDGOSUV' for _ in self.direct.types):
                real = False
                if self.dtype is None:
                    self.dtype = np.dtype(np.complex128)
            elif all(_[3] in 'efdgEFDGOSUV' for _ in self.direct.types):
                if self.dtype is None:
                    self.dtype = np.dtype(np.float64)
            if real:
                self._set_flags('real')
            self._set_flags('inplace')
            self._set_flags('square')
            self._set_flags('separable')
            if self.direct is np.negative:
                self._set_flags('linear')

        if self.flags.inplace:
            aligned = max(self.flags.aligned_input,
                          self.flags.aligned_output)
            contiguous = max(self.flags.contiguous_input,
                             self.flags.contiguous_output)
            self._set_flags({'aligned_input': aligned,
                             'aligned_output': aligned,
                             'contiguous_input': contiguous,
                             'contiguous_output': contiguous})

    def _init_rules(self):
        """ Translate flags into rules. """
        if self.rules is None:
            self.rules = {}

        if self.flags.real:
            self.set_rule('C', '.')
        if self.flags.symmetric:
            self.set_rule('T', '.')
        if self.flags.hermitian:
            self.set_rule('H', '.')
        if self.flags.involutary:
            self.set_rule('I', '.')

        self.set_rule('I,.', '1', CompositionOperator)
        if self.flags.orthogonal:
            self.set_rule('T,.', '1', CompositionOperator)
        if self.flags.unitary:
            self.set_rule('H,.', '1', CompositionOperator)
        if self.flags.idempotent:
            self.set_rule('.,.', '.', CompositionOperator)
        if self.flags.involutary:
            self.set_rule('.,.', '1', CompositionOperator)

    def _init_inout(self, attrin, attrout, classin, classout, commin, commout,
                    reshapein, reshapeout, shapein, shapeout, toshapein,
                    toshapeout, validatein, validateout):
        """
        Set methods and attributes dealing with the input and output handling.
        """

        if isinstance(attrin, (dict, types.FunctionType, types.MethodType)):
            if not isinstance(attrin, dict) or len(attrin) > 0:
                self.attrin = attrin
        else:
            raise TypeError(
                "The 'attrin' keyword should be a dictionary or a function.")
        if isinstance(attrout, (dict, types.FunctionType, types.MethodType)):
            if not isinstance(attrout, dict) or len(attrout) > 0:
                self.attrout = attrout
        else:
            raise TypeError(
                "The 'attrout' keyword should be a dictionary or a function.")
        if type(classin) is type and issubclass(classin, np.ndarray):
            self.classin = classin
        elif classin is not None:
            raise TypeError(
                "The 'classin' keyword is not an ndarray subclass.")
        if type(classout) is type and issubclass(classout, np.ndarray):
            self.classout = classout
        elif classout is not None:
            raise TypeError(
                "The 'classout' keyword is not an ndarray subclass.")
        if commin is not None:
            self.commin = commin
        if commout is not None:
            self.commout = commout
        if reshapein is not None:
            self.reshapein = reshapein
        if reshapeout is not None:
            self.reshapeout = reshapeout
        if toshapein is not None:
            self.toshapein = toshapein
        if toshapeout is not None:
            self.toshapeout = toshapeout
        if validatein is not None:
            self.validatein = validatein
        if validateout is not None:
            self.validateout = validateout

        self.shapein = tointtuple(shapein)
        self.shapeout = tointtuple(shapeout)
        if self.shapein is not None:
            shapeout = tointtuple(self.reshapein(self.shapein))
            if self.shapeout is None:
                self.shapeout = shapeout
            else:
                self.validateout(shapeout)
        if self.shapeout is not None:
            shapein = tointtuple(self.reshapeout(self.shapeout))
            if self.shapein is None:
                self.shapein = shapein
            else:
                self.validatein(shapein)
        if self.shapein is not None:
            self.validatein(self.shapein)
        if self.shapeout is not None:
            self.validateout(self.shapeout)

        if self.shapein is not None and self.shapeout is not None:
            self._set_flags(square=self.shapein == self.shapeout)

        if self.flags.square:
            if self.shapein is None:
                self.shapein = self.shapeout
            else:
                self.shapeout = self.shapein
            self.reshapein = lambda x: x
            self.reshapeout = self.reshapein
            self.toshapeout = self.toshapein
            self.validateout = self.validatein

        if self.shapein is not None:
            try:
                del self.toshapein
            except AttributeError:
                pass
        if self.shapeout is not None:
            try:
                del self.toshapeout
            except AttributeError:
                pass

        flag_is = 'explicit' if self.shapein is not None else 'implicit' if \
            self.reshapeout != Operator.reshapeout.__get__(self, type(self)) \
            else 'unconstrained'
        flag_os = 'explicit' if self.shapeout is not None else 'implicit' \
            if self.reshapein != Operator.reshapein.__get__(self, type(self)) \
            else 'unconstrained'
        self._set_flags(shape_input=flag_is, shape_output=flag_os)

        if self.flags.shape_input == 'explicit':
            self.reshapeout = Operator.reshapeout.__get__(self, type(self))
            self.validatein = Operator.validatein.__get__(self, type(self))
        if self.flags.shape_output == 'explicit':
            self.reshapein = Operator.reshapein.__get__(self, type(self))
            self.validateout = Operator.validateout.__get__(self, type(self))

    def _init_name(self, name):
        """ Set operator's __name__ attribute. """
        if name is None:
            if self.__name__ is not None:
                return
            if type(self) is not Operator:
                name = type(self).__name__
            elif self.direct is not None and self.direct.__name__ not in \
                    ('<lambda>', 'direct'):
                name = self.direct.__name__
            else:
                name = 'Operator'
        self.__name__ = name

    def _reset(self, **keywords_):
        """
        Use this method with cautious: the operator's flags are carried over
        unless the 'flag' keyword is specified. It may lead to inconsistencies.

        """
        keywords = dict((k, v)
                        for k, v in self.__dict__.items()
                        if k in OPERATOR_ATTRIBUTES)
        keywords.update(keywords_)

        # reset attributes
        for attr in OPERATOR_ATTRIBUTES + ['_C', '_T', '_H', '_I']:
            if attr in self.__dict__:
                del self.__dict__[attr]

        # re-init operator with new attributes
        Operator.__init__(self, **keywords)

    def _set_flags(self, flags=None, **keywords):
        """ Set flags to an Operator. """
        if isinstance(flags, Flags) and len(keywords) == 0:
            self.flags = flags
            return
        flags = self.validate_flags(flags, **keywords)
        for flag in ('hermitian', 'orthogonal', 'symmetric', 'unitary'):
            if flags.get(flag, False):
                flags['linear'] = True
                break
        self.flags = self.flags._replace(**flags)

    def _validate_arguments(self, input, output):
        """
        Return the input and output as ndarray instances.
        If required, allocate the output.
        """
        input = np.array(input, copy=False)
        dtype = self._find_common_type([input.dtype, self.dtype])

        input_ = None
        output_ = None

        # if the input is not compatible, copy it into a buffer from the pool
        if input.dtype != dtype or not iscompatible(input, input.shape, dtype,
           self.flags.aligned_input, self.flags.contiguous_input):
            if output is not None and self.flags.inplace and iscompatible(
               output, input.shape, dtype, self.flags.aligned_input,
               self.flags.contiguous_input):
                buf = output
            else:
                input_ = _pool.extract(input.shape, dtype,
                                       self.flags.aligned_input,
                                       self.flags.contiguous_input)
                buf = input_
            input, input[...] = _pool.view(buf, input.shape, dtype), input

        # check compatibility of provided output
        if output is not None:
            if not isinstance(output, np.ndarray):
                raise TypeError('The output argument is not an ndarray.')
            output = output.view(np.ndarray)
            if output.dtype != dtype:
                raise ValueError(
                    "The output has an invalid dtype '{0}'. Expected dtype is "
                    "'{1}'.".format(output.dtype, dtype))

            # if the output does not fulfill the operator's alignment &
            # contiguity requirements, or if the operator is out-of-place and
            # an in-place operation is required, let's use a temporary buffer
            if not iscompatible(output, output.shape, dtype,
               self.flags.aligned_output, self.flags.contiguous_output) or \
               isalias(input, output) and not self.flags.inplace:
                output_ = _pool.extract(
                    output.shape, dtype, self.flags.aligned_output,
                    self.flags.contiguous_output)
                output = _pool.view(output_, output.shape, dtype)
            shapeout = output.shape
        else:
            shapeout = None

        shapein, shapeout = self._validate_shapes(input.shape, shapeout)

        # if the output is not provided, allocate it
        if output is None:
            if self.flags.shape_input == 'implicit' and \
               self.flags.shape_output == 'unconstrained':
                raise ValueError(
                    'The output shape of an implicit input shape and unconstra'
                    'ined output shape operator cannot be inferred.')
            if shapeout is None:
                shapeout = input.shape
            output = empty(shapeout, dtype, description=
                           "for {0}'s output.".format(self.__name__))
        return input, input_, output, output_

    @staticmethod
    def validate_flags(flags, **keywords):
        """ Return flags as a dictionary. """
        if flags is None:
            return keywords
        if isinstance(flags, dict):
            flags = flags.copy()
        elif isinstance(flags, Flags):
            flags = dict((k, v) for k, v in zip(Flags._fields, flags))
        elif isinstance(flags, (list, tuple, str)):
            if isinstance(flags, str):
                flags = [f.strip() for f in flags.split(',')]
            flags = dict((f, True) for f in flags)
        else:
            raise TypeError("The operator flags have an invalid type '{0}'.".
                            format(flags))
        flags.update(keywords)
        if any(not isinstance(f, str) for f in flags):
            raise TypeError("Invalid type for the operator flags: {0}."
                            .format(flags))
        if any(f not in Flags._fields for f in flags):
            raise ValueError(
                "Invalid operator flags '{0}'. The properties must be one of t"
                "he following: ".format(flags.keys()) + strenum(
                Flags._fields) + '.')
        return flags

    def _validate_shapes(self, shapein, shapeout):
        """
        Validate that the arguments shapein and shapeout are compatible with
        the input and output shapes of the operator. The arguments can be None
        to signify that they are unknown. The input and output shapes of the
        operator (inferred from the known arguments if necessary) are then
        returned.
        This method should be used with initialised operators.

        """
        shapein = tointtuple(shapein)
        if shapein is not None:
            self.validatein(shapein)
        if self.flags.shape_output == 'explicit':
            shapeout_ = self.shapeout
        elif self.flags.shape_output == 'unconstrained' or shapein is None:
            shapeout_ = None
        else:
            shapeout_ = tointtuple(self.reshapein(shapein))
            self.validateout(shapeout_)

        shapeout = tointtuple(shapeout)
        if shapeout is not None:
            self.validateout(shapeout)
        if self.flags.shape_input == 'explicit':
            shapein_ = self.shapein
        elif self.flags.shape_input == 'unconstrained' or shapeout is None:
            shapein_ = None
        else:
            shapein_ = tointtuple(self.reshapeout(shapeout))
            self.validatein(shapein_)

        if None not in (shapein, shapein_) and shapein != shapein_:
            raise ValueError(
                "The specified input shape '{0}' is incompatible with the expe"
                "cted one '{1}'.".format(shapein, shapein_))
        if None not in (shapeout, shapeout_) and shapeout != shapeout_:
            raise ValueError(
                "The specified output shape '{0}' is incompatible with the exp"
                "ected one '{1}'.".format(shapeout, shapeout_))

        return (first_is_not([shapein, shapein_], None),
                first_is_not([shapeout, shapeout_], None))

    def __truediv__(self, other):
        return MultiplicationOperator([self,
                                       po.nonlinear.PowerOperator(-1)(other)])
    __div__ = __truediv__

    def __rtruediv__(self, other):
        return MultiplicationOperator([other,
                                       po.nonlinear.PowerOperator(-1)(self)])
    __rdiv__ = __rtruediv__

    def __mul__(self, other):
        if isinstance(other, (Variable, VariableTranspose)):
            return other.__rmul__(self)
        if (self.flags.linear and
            not isscalarlike(other) and
            isinstance(other, (np.ndarray, list, tuple)) and
            not isinstance(other, np.matrix)):
                return self(other)
        try:
            other = asoperator(other, constant=not self.flags.linear)
        except TypeError:
            return NotImplemented
        if not self.flags.linear or not other.flags.linear:
            return MultiplicationOperator([self, other])
        # ensure that A * A is A if A is idempotent
        if self.flags.idempotent and self is other:
            return self
        return CompositionOperator([self, other])

    def __rmul__(self, other):
        if (self.flags.linear and
            not isscalarlike(other) and
            isinstance(other, (np.ndarray, list, tuple)) and
            not isinstance(other, np.matrix)):
                return self.T(other)
        try:
            other = asoperator(other, constant=not self.flags.linear)
        except TypeError:
            return NotImplemented
        if not self.flags.linear or not other.flags.linear:
            return MultiplicationOperator([other, self])
        return CompositionOperator([other, self])

    def __pow__(self, n):
        if not self.flags.linear:
            return po.nonlinear.PowerOperator(n)(self)
        if not np.allclose(n, np.round(n)):
            raise ValueError("The exponent '{0}' is not an integer.".format(n))
        if n == -1:
            return self.I
        if n == 0:
            return IdentityOperator(shapein=self.shapein)
        if n == 1:
            return self
        if n > 0:
            return CompositionOperator(n * [self])
        return CompositionOperator((-n) * [self.I])

    def __add__(self, other):
        return AdditionOperator([self, other])

    def __radd__(self, other):
        return AdditionOperator([other, self])

    def __sub__(self, other):
        return AdditionOperator([self, -other])

    def __rsub__(self, other):
        return AdditionOperator([other, -self])

    def __neg__(self):
        return HomothetyOperator(-1) * self

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        d1 = self.__dict__.copy()
        d2 = other.__dict__.copy()
        for k in 'rules', '_C', '_T', '_H', '_I', '_D':
            if k in d1:
                del d1[k]
            if k in d2:
                del d2[k]
        return all_eq(d1, d2)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        if self.__name__ is None:
            return type(self).__name__ + ' [not initialized]'
        if self.flags.linear and (self.shapein is not None or
                                  self.shapeout is not None):
            shapein = '?' if self.shapein is None else strshape(self.shapein)
            shapeout = '?' if self.shapeout is None else \
                strshape(self.shapeout)
            if self.flags.square and self.shapein is not None and \
               len(self.shapein) > 1:
                s = shapein + '²'
            else:
                s = shapeout + 'x' + shapein
            s += ' '
        else:
            s = ''
        name = self.__name__
        if name != 'Operator':
            name = name.replace('Operator', '')
        s += name.lower()
        return s

    def __repr__(self):
        if self.__name__ is None:
            return type(self).__name__ + ' [not initialized]'

        a = []
        init = getattr(self, '__init_original__', self.__init__)
        vars, args, keywords, defaults = inspect.getargspec(init)
        if defaults is None:
            defaults = []
        else:
            defaults = list(defaults)

        #XXX it would be better to walk the Operator's hirarchy
        # to grab all keywords.
        if 'shapein' not in vars:
            vars.append('shapein')
            defaults.append(None)
        if 'shapeout' not in vars:
            vars.append('shapeout')
            defaults.append(None)

        for ivar, var in enumerate(vars):
            if var in ('flags', 'self'):
                continue
            if var == 'shapeout' and self.flags.shape_output == 'implicit':
                continue
            if var == 'shapein' and self.flags.shape_input == 'implicit':
                continue
            if var == 'reshapeout' and self.flags.square and \
               self.flags.shape_input == 'implicit':
                continue

            val = getattr(self, var, None)
            if isinstance(val, types.MethodType):
                continue
            nargs = len(vars) - len(defaults)
            if ivar >= nargs:
                try:
                    if val == defaults[ivar - nargs]:
                        continue
                except:
                    if val is defaults[ivar - nargs]:
                        continue
            if var == 'reshapein' and self.flags.square and \
               self.flags.shape_output == 'implicit':
                s = 'lambda x:x'
            elif var in ('commin', 'commout'):
                if val is MPI.COMM_WORLD:
                    s = 'MPI.COMM_WORLD'
                elif val is MPI.COMM_SELF:
                    s = 'MPI.COMM_SELF'
                else:
                    s = str(val)
            elif isinstance(val, Operator):
                s = 'Operator()'
            elif type(val) is type:
                s = val.__module__ + '.' + val.__name__
            elif var in ['shapein', 'shapeout']:
                s = strshape(val)
            elif isinstance(val, np.ndarray) and val.ndim == 0:
                s = repr(val[()])
            elif isinstance(val, np.ndarray):
                s = 'array' if type(val) is np.ndarray else type(val).__name__
                s += '(' + val.ndim * '['
                s += str(val.flat[0])
                if val.size > 1:
                    s += ', ' if val.size == 2 else ', ..., '
                    s += str(val.flat[-1])
                s += val.ndim * ']' + ', dtype={0})'.format(val.dtype)
            elif var == 'dtype':
                s = str(val)
            else:
                s = repr(val)

            if ivar < nargs:
                a += [s]
            else:
                a += [var + '=' + s]
        return self.__name__ + '(' + ', '.join(a) + ')'


class DeletedOperator(Operator):
    def __init__(self):
        raise NotImplementedError('A DeletedOperator cannot be instantiated.')

    __name__ = 'DeletedOperator'


@real
@square
@symmetric
@idempotent
@involutary
@update_output
class CopyOperator(Operator):
    """
    Copy operator.

    Unlike IdentityOperator, this is an out-of-place operator.

    """
    def direct(self, input, output, operation=operation_assignment):
        operation(output, input)


class CompositeOperator(Operator):
    """
    Abstract class for handling a list of operands.

    Attributes
    ----------
    operands : list of Operators
        List of operands.

    Methods
    -------
    can_morph : boolean method
        If the composite operator has only one operand (being the argument
        itself or being the result of simplifications by binary rules),
        this method specifues if the composite should morph into its operand.
        Default is False.

    Notes
    -----
    Composites can morph into their single operand if the attribute
    'morph_single_operand' is set to True. As a consequence, one should make
    sure to return right after the call in the parent __init__ method.

    """
    def __init__(self, operands, dtype=None, **keywords):
        self._validate_comm(operands)
        if dtype is None:
            dtype = self._find_common_type(o.dtype for o in operands)
        self.operands = operands
        Operator.__init__(self, dtype=dtype, **keywords)
        self.propagate_commin(self.commin)
        self.propagate_commout(self.commout)

    morph_single_operand = True

    @property
    def nbytes(self):
        d = dict((id(_), _) for _ in self.operands)
        unique = set(d.keys())
        return sum(d[_].nbytes for _ in unique)

    def propagate_attributes(self, cls, attr):
        return self.operands[0].propagate_attributes(cls, attr)

    def propagate_commin(self, commin):
        if commin is None:
            return self
        self.commin = commin
        for i, op in enumerate(self.operands):
            self.operands[i] = op.propagate_commin(commin)
        return self

    def propagate_commout(self, commout):
        if commout is None:
            return self
        self.commout = commout
        for i, op in enumerate(self.operands):
            self.operands[i] = op.propagate_commout(commout)
        return self

    def _apply_rules(self, ops):
        return ops

    def _validate_operands(self, operands, constant=False):
        if not isinstance(operands, (list, tuple, types.GeneratorType)):
            operands = [operands]
        return [asoperator(op, constant=constant) for op in operands]

    def _validate_comm(self, operands):
        comms = [op.commin for op in operands if op.commin is not None]
        if len(set(id(c) for c in comms)) > 1:
            raise ValueError('The input MPI communicators are incompatible.')
        comms = [op.commout for op in operands if op.commout is not None]
        if len(set(id(c) for c in comms)) > 1:
            raise ValueError('The output MPI communicators are incompatible.')
        return operands

    def __str__(self):
        if isinstance(self, AdditionOperator):
            op = ' + '
        elif isinstance(self, MultiplicationOperator):
            op = u' \u00d7 '
        elif isinstance(self, (BlockDiagonalOperator, BlockSliceOperator)):
            op = u' \u2295 '
        else:
            op = ' * '

        # parentheses for AdditionOperator and BlockDiagonalOperator
        operands = ['({0})'.format(o) if isinstance(o, (AdditionOperator,
                    BlockDiagonalOperator)) else str(o) for o in self.operands]

        # some special cases
        if isinstance(self, BlockDiagonalOperator) and len(operands) > 2:
            operands = [operands[0], '...', operands[-1]]
        elif isinstance(self, CompositionOperator) and \
                isinstance(self.operands[0], HomothetyOperator):
            # remove trailing 'I'
            operands[0] = operands[0][:-1]
            if self.operands[0].data == -1:
                operands[0] += '1'

        op = op.join(operands)
        if sys.version_info.major == 2:
            op = op.encode('utf-8')
        return op

    def __repr__(self):
        r = self.__name__ + '(['
        rops = [repr(op) for op in self.operands]
        components = []
        for i, rop in enumerate(rops):
            if i != len(rops) - 1:
                rop += ','
            components.extend(rop.split('\n'))
        r += '\n    '+'\n    '.join(components) + '])'
        return r


class CommutativeCompositeOperator(CompositeOperator):
    """
    Abstract class for commutative composite operators, such as the addition or
    the element-wise multiplication.

    """
    def __init__(self, operands, operation=None, **keywords):
        keywords = self._get_attributes(operands, **keywords)
        operands = self._apply_rules(operands)
        if len(operands) == 1 and self.morph_single_operand:
            operands[0].copy(self)
            self._reset(**keywords)
            return
        CompositeOperator.__init__(self, operands, **keywords)
        self.set_rule(('.', Operator), lambda s, o: type(s)(s.operands + [o]),
                      type(self))
        self.set_rule(('.', type(self)), lambda s, o:
                      type(s)(s.operands + o.operands), type(self))
        self.operation = operation

    def direct(self, input, output):
        operands = list(self.operands)
        assert len(operands) > 1

        # we need a temporary buffer if all operands can do inplace reductions
        # except no more than one, which is move as first operand
        try:
            ir = [o.flags.update_output for o in operands]
            index = ir.index(False)
            operands[0], operands[index] = operands[index], operands[0]
            need_temporary = ir.count(False) > 1
        except ValueError:
            need_temporary = False

        operands[0].direct(input, output)
        ii = 0
        with _pool.get_if(need_temporary, output.shape, output.dtype) as buf:
            for op in operands[1:]:
                if op.flags.update_output:
                    op.direct(input, output, operation=self.operation)
                else:
                    op.direct(input, buf)
                    self.operation(output, buf)
                ii += 1

    def propagate_attributes(self, cls, attr):
        return Operator.propagate_attributes(self, cls, attr)

    def _apply_rules(self, ops):
        if po.rules.rule_manager['none']:
            return ops

        if DEBUG:
            strcls = type(self).__name__.upper()[:-8]

            def print_operands():
                print()
                print(len(strcls) * '=' + '=========')
                print(strcls + ' OPERANDS')
                print(len(strcls) * '=' + '=========')
                for i, op in enumerate(ops):
                    print('{0}: {1!r}'.format(i, op))
            print_operands()

        if len(ops) <= 1:
            if DEBUG:
                print('OUT (only one operand)')
                print()
            return ops
        i = 0
        while i < len(ops):
            if type(self) not in ops[i].rules:
                i += 1
                continue
            j = 0
            consumed = False
            while j < len(ops):
                if j != i:
                    for rule in ops[i].rules[type(self)]:
                        if DEBUG:
                            print("({0}, {1}): testing rule '{2}'".
                                  format(i, j, rule))
                        new_ops = rule(ops[i], ops[j])
                        if new_ops is None:
                            continue
                        if DEBUG:
                            print('Because of rule {0}:'.format(rule))
                            print('     MERGING ({0}, {1}) into {2!s} ~ {2!r}'.
                                  format(i, j, new_ops))
                        del ops[j]
                        if j < i:
                            i -= 1
                        ops[i] = new_ops
                        if DEBUG:
                            print_operands()
                        consumed = True
                        break
                    if consumed:
                        break
                if consumed:
                    break
                j += 1
            if consumed:
                continue
            i += 1

        # move this up to avoid creations of temporaries
        i = [i for i, o in enumerate(ops) if isinstance(o, HomothetyOperator)]
        if len(i) > 0:
            ops.insert(0, ops[i[0]])
            del ops[i[0]+1]
            if ops[0].data == 0 and len(ops) > 1:
                del ops[0]
        return ops

    @classmethod
    def _get_attributes(cls, operands, **keywords):
        attr = {
            'attrin': first_is_not((o.attrin for o in operands), None),
            'attrout': first_is_not((o.attrout for o in operands), None),
            'classin': first_is_not((o.classin for o in operands), None),
            'classout': first_is_not((o.classout for o in operands), None),
            'commin': first_is_not((o.commin for o in operands), None),
            'commout': first_is_not((o.commout for o in operands), None),
            'dtype': cls._find_common_type(o.dtype for o in operands),
            'flags': cls._merge_flags(operands),
            'reshapein': cls._merge_reshapein(operands),
            'reshapeout': cls._merge_reshapeout(operands),
            'shapein': cls._merge_shape((o.shapein for o in operands), 'in'),
            'shapeout': cls._merge_shape((o.shapeout for o in operands),
                                         'out'),
            'toshapein': first_is_not((o.toshapein for o in operands), None),
            'toshapeout': first_is_not((o.toshapeout for o in operands), None),
            'validatein': first_is_not((o.validatein for o in operands), None),
            'validateout': first_is_not((o.validateout for o in operands),
                                        None)}
        for k, v in keywords.items():
            if k is not 'flags':
                attr[k] = v
        attr['flags'].update(
            Operator.validate_flags(keywords.get('flags', {})))
        return attr

    @staticmethod
    def _merge_flags(operands):
        return {
            'real': all(o.flags.real for o in operands),
            'aligned_input': max(o.flags.aligned_input for o in operands),
            'aligned_output': max(o.flags.aligned_output for o in operands),
            'contiguous_input': any(o.flags.contiguous_input
                                    for o in operands),
            'contiguous_output': any(o.flags.contiguous_output
                                     for o in operands)}

    @staticmethod
    def _merge_reshapein(operands):
        if any(o.flags.shape_output == 'explicit' for o in operands):
            return None
        if all(o.flags.shape_output == 'unconstrained' for o in operands):
            return None
        return first_is_not((o.reshapein for o in operands
                             if o.flags.shape_output == 'implicit'), None)

    @staticmethod
    def _merge_reshapeout(operands):
        if any(o.flags.shape_input == 'explicit' for o in operands):
            return None
        if all(o.flags.shape_input == 'unconstrained' for o in operands):
            return None
        return first_is_not((o.reshapeout for o in operands
                             if o.flags.shape_input == 'implicit'), None)

    @staticmethod
    def _merge_shape(shapes, inout):
        shapes = [s for s in shapes if s is not None]
        if len(shapes) == 0:
            return None
        if any(s != shapes[0] for s in shapes):
            raise ValueError('The {0}put shapes are incompatible: {1}.'.format(
                             inout, strenum(shapes, 'and')))
        return shapes[0]


class AdditionOperator(CommutativeCompositeOperator):
    """
    Class for operator addition

    If at least one of the input already is the result of an addition,
    a flattened list of operators is created by associativity, to simplify
    reduction.

    """
    def __init__(self, operands, **keywords):
        operands = self._validate_operands(operands)
        CommutativeCompositeOperator.__init__(self, operands, operator.iadd,
                                              **keywords)
        if not isinstance(self, CommutativeCompositeOperator):
            return
        self.set_rule('C', lambda s: type(s)([m.C for m in s.operands]))
        self.set_rule('T', lambda s: type(s)([m.T for m in s.operands]))
        self.set_rule('H', lambda s: type(s)([m.H for m in s.operands]))

    @staticmethod
    def _merge_flags(operands):
        flags = CommutativeCompositeOperator._merge_flags(operands)
        flags.update({
            'linear': all(op.flags.linear for op in operands),
            'separable': all(o.flags.separable for o in operands),
            'square': any(o.flags.square for o in operands),
            'symmetric': all(op.flags.symmetric for op in operands),
            'hermitian': all(op.flags.symmetric for op in operands)})
        return flags


class MultiplicationOperator(CommutativeCompositeOperator):
    """
    Class for Hadamard (element-wise) multiplication of operators.

    If at least one of the input already is the result of an multiplication,
    a flattened list of operators is created by associativity, to simplify
    reduction.

    """
    def __init__(self, operands, **keywords):
        operands = self._validate_operands(operands, constant=True)
        CommutativeCompositeOperator.__init__(self, operands, operator.imul,
                                              **keywords)
        if not isinstance(self, CommutativeCompositeOperator):
            return
        self.set_rule('C', lambda s: type(s)([m.C for m in s.operands]))

    @staticmethod
    def _merge_flags(operands):
        flags = CommutativeCompositeOperator._merge_flags(operands)
        flags.update({
            'separable': all(o.flags.separable for o in operands),
            'square': any(o.flags.square for o in operands)})
        return flags


@square
class BlockSliceOperator(CommutativeCompositeOperator):
    """
    Class for multiple disjoint slices.

    The elements of the input not included in the slices are copied over to
    the output. This is due to fact that it is not easy to derive
    the complement of a set of slices. To set those values to zeros, you might
    use MaskOperator or write a custom operator.
    Currently, there is no check to verify that the slices are disjoint.
    Non-disjoint slices can lead to unexpected results.

    Examples
    --------
    >>> op = BlockSliceOperator(HomothetyOperator(3), slice(None,None,2))
    >>> op(np.ones(6))
    array([ 3.,  1.,  3.,  1.,  3.,  1.])

    >>> op = BlockSliceOperator([ConstantOperator(1), ConstantOperator(2)],
    ...                         ([slice(0, 2), slice(0, 2)],
    ...                          [slice(2, 4), slice(2, 4)]))
    >>> op(np.zeros((4,4)))
    array([[ 1.,  1.,  0.,  0.],
           [ 1.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  2.],
           [ 0.,  0.,  2.,  2.]])

    """
    def __init__(self, operands, slices, **keywords):
        operands = self._validate_operands(operands)
        if any(not op.flags.square and op.flags.shape_output != 'unconstrained'
               for op in operands):
            raise ValueError('Input operands must be square.')
        if not isinstance(slices, (list, tuple, types.GeneratorType, slice)):
            raise TypeError('Invalid input slices.')
        if isinstance(slices, slice):
            slices = (slices,)
        else:
            slices = tuple(slices)
        if len(operands) != len(slices):
            raise ValueError(
                "The number of slices '{0}' is not equal to the number of oper"
                "ands '{1}'.".format(len(slices), len(operands)))

        CommutativeCompositeOperator.__init__(self, operands, **keywords)
        self.slices = slices
        self.set_rule('C', lambda s: BlockSliceOperator(
                      [op.C for op in s.operands], s.slices))
        self.set_rule('T', lambda s: BlockSliceOperator(
                      [op.T for op in s.operands], s.slices))
        self.set_rule('H', lambda s: BlockSliceOperator(
                      [op.H for op in s.operands], s.slices))
        self.set_rule(('.', HomothetyOperator),
                      lambda s, o: BlockSliceOperator(
                          [o.data * op for op in s.operands], s.slices),
                      CompositionOperator)

    morph_single_operand = False

    def direct(self, input, output):
        if not isalias(input, output):
            output[...] = input
        for s, op in zip(self.slices, self.operands):
            i = input[s]
            o = output[s]
            with _pool.copy_if(i, op.flags.aligned_input,
                               op.flags.contiguous_input) as i:
                with _pool.copy_if(o, op.flags.aligned_output,
                                   op.flags.contiguous_output) as o:
                    op.direct(i, o)

    @classmethod
    def _get_attributes(cls, operands, **keywords):
        attr = {
            'dtype': cls._find_common_type(o.dtype for o in operands),
            'flags': cls._merge_flags(operands),
        }
        for k, v in keywords.items():
            if k is not 'flags':
                attr[k] = v
        attr['flags'].update(
            Operator.validate_flags(keywords.get('flags', {})))
        return attr

    @staticmethod
    def _merge_flags(operands):
        flags = CommutativeCompositeOperator._merge_flags(operands)
        flags.update({
            'linear': all(op.flags.linear for op in operands),
            'symmetric': all(op.flags.symmetric for op in operands),
            'hermitian': all(op.flags.hermitian for op in operands),
            'inplace': all(op.flags.inplace for op in operands)})
        return flags


class NonCommutativeCompositeOperator(CompositeOperator):
    """
    Abstract class for non-commutative composite operators, such as
    the composition.

    """
    def _apply_rules(self, ops):
        if po.rules.rule_manager['none']:
            return ops

        if DEBUG:
            def print_rules(i, rules):
                print('Rules for ({0}, {1}):'.format(i, i+1))
                for i, r in enumerate(rules):
                    print('    {0}: {1}'.format(i, r))
                print()

            def print_operands():
                print()
                print('====================')
                print('COMPOSITION OPERANDS')
                print('====================')
                for i, op in enumerate(ops):
                    print('{0}: {1!r}'.format(i, op))
            import pdb
            print()
            print()
            print()
            pdb.traceback.print_stack()
            print_operands()

        if len(ops) <= 1:
            if DEBUG:
                print('OUT (only one operand)')
                print()
            return ops

        # Get the NonCommutativeCompositeOperator direct subclass
        cls = type(self).__mro__[-5]

        i = len(ops) - 2
        # loop over the len(ops)-1 pairs of operands
        while i >= 0:
            o1 = ops[i]
            o2 = ops[i+1]
            rules1 = o1.rules[cls]['left'] if cls in o1.rules else []
            rules2 = o2.rules[cls]['right'] if cls in o2.rules else []

            def key_rule(x):
                if isinstance(x.other, str):
                    return 0
                if x.reference == 0:
                    return 1000 - len(type(o1).__mro__) - len(x.other.__mro__)
                return 1000 - len(x.other.__mro__) - len(type(o2).__mro__)

            rules = rules1 + rules2
            rules.sort(key=key_rule)

            if DEBUG > 1:
                print_rules(i, rules)
            consumed = False
            for rule in rules:
                new_ops = rule(o1, o2)
                if new_ops is None:
                    continue
                consumed = True
                if DEBUG:
                    print('Because of rule {0}:'.format(rule))
                if isinstance(new_ops, tuple):
                    if len(new_ops) != 2:
                        raise NotImplementedError()
                    ops[i], ops[i+1] = new_ops
                    if DEBUG:
                        print('    DOUBLE CHANGE: {0} into {1}'.format(
                              i, new_ops[0]))
                        print('    DOUBLE CHANGE: {0} into {1}'.format(
                              i+1, new_ops[1]))
                        print_operands()
                    i += 1
                    break
                if DEBUG:
                    print('     MERGING ({0}, {1}) into {2!s} ~ {2!r}'.format(
                          i, i+1, new_ops))
                cls._merge(new_ops, o1, o2)
                del ops[i+1]
                ops[i] = new_ops
                if DEBUG:
                    print_operands()
                break

            if consumed and i < len(ops) - 1:
                continue

            i -= 1

        if DEBUG:
            print('OUT', end=' ')
            if len(ops) == 1:
                print('(only one operand)')
            else:
                print('(because of rule exhaustion)')
            print()
            print()

        return ops


@inplace
class CompositionOperator(NonCommutativeCompositeOperator):
    """
    Class handling operator composition.

    If at least one of the input already is the result of a composition,
    a flattened list of operators is created by associativity, to simplify
    reduction.

    """
    def __init__(self, operands, **keywords):
        operands = self._validate_operands(operands)
        operands = self._apply_rules(operands)
        if len(operands) == 1 and self.morph_single_operand:
            operands[0].copy(self)
            self._reset(**keywords)
            return
        keywords = self._get_attributes(operands, **keywords)
        self._info = {}
        NonCommutativeCompositeOperator.__init__(self, operands, **keywords)
        self.set_rule('C', lambda s: CompositionOperator(
                      [m.C for m in s.operands]))
        self.set_rule('T', lambda s: CompositionOperator(
                      [m.T for m in s.operands[::-1]]))
        self.set_rule('H', lambda s: CompositionOperator(
                      [m.H for m in s.operands[::-1]]))
        self.set_rule('I', lambda s: CompositionOperator(
                      [m.I for m in s.operands[::-1]]))
        self.set_rule('IC', lambda s: CompositionOperator(
                      [m.I.C for m in s.operands[::-1]]))
        self.set_rule('IT', lambda s: CompositionOperator(
                      [m.I.T for m in s.operands]))
        self.set_rule('IH', lambda s: CompositionOperator(
                      [m.I.H for m in s.operands]))
        self.set_rule(('.', CompositionOperator), lambda s, o:
                      CompositionOperator(s.operands + o.operands),
                      CompositionOperator)
        self.set_rule(('.', Operator), lambda s, o: CompositionOperator(
                      s.operands + [o]), CompositionOperator)
        self.set_rule((Operator, '.'), lambda o, s: CompositionOperator(
                      [o] + s.operands), CompositionOperator)

    def direct(self, input, output, operation=operation_assignment,
               preserve_input=True):

        preserve_input &= not isalias(input, output)
        preserve_output = operation is not operation_assignment

        shapeouts, dtypes, ninplaces, bufsizes, aligneds, contiguouss = \
            self._get_info(input, output, preserve_input)

        i = i_ = input
        if isalias(input, output):
            o_ = output if output.nbytes > input.nbytes else input
        else:
            o_ = output
        iop = len(self.operands) - 1
        ngroups = len(ninplaces)
        reuse_output = True

        # outer loop over groups of operators
        for igroup, (ninplace, bufsize, aligned, contiguous) in renumerate(
                zip(ninplaces, bufsizes, aligneds, contiguouss)):

            if igroup != ngroups - 1:

                # get output for the current outplace operator if possible
                reuse_output = not preserve_output and (igroup % 2 == 0) and \
                    iscompatible(output, bufsize, np.int8, aligned,
                                 contiguous) and not isalias(output, i) or \
                    igroup == 0
                if reuse_output:
                    o_ = output
                else:
                    o_ = _pool.extract(bufsize, np.int8, aligned, contiguous)
                    _pool.add(output)
                o = _pool.view(o_, shapeouts[iop], dtypes[iop])
                op = self.operands[iop]

                # perform out-of place operation
                if iop == 0 and self.flags.update_output:
                    op.direct(i, o, operation=operation)
                else:
                    op.direct(i, o)
                iop -= 1

                # set the input buffer back in the pool
                if (igroup < ngroups - 2 or not preserve_input) and \
                   not isalias(i_, output):
                    _pool.add(i_)
                i = o
                i_ = o_

            # loop over inplace operations
            for n in range(ninplace):
                o = _pool.view(o_, shapeouts[iop], dtypes[iop])
                op = self.operands[iop]
                op.direct(i, o)
                i = o
                iop -= 1

            # get the output out of the pool
            if not reuse_output:
                _pool.remove(output)

        if ngroups >= 2 and not preserve_input and \
           not isalias(input, output):
            _pool.remove(input)

    def propagate_attributes(self, cls, attr):
        for op in reversed(self.operands):
            cls = op.propagate_attributes(cls, attr)
        return cls

    def propagate_commin(self, commin):
        if commin is None:
            return self
        self.commin = commin
        for i, op in reversed(list(enumerate(self.operands))):
            if op.commin is not None:
                commin = op.commout
            else:
                op = op.propagate_commin(commin)
                self.operands[i] = op
                commin = op.commout or commin
        return self

    def propagate_commout(self, commout):
        if commout is None:
            return self
        self.commout = commout
        for i, op in enumerate(self.operands):
            if op.commout is not None:
                commout = op.commin
            else:
                op = op.propagate_commout(commout)
                self.operands[i] = op
                commout = op.commin or commout
        return self

    def _apply_rules(self, ops):
        if po.rules.rule_manager['none']:
            return ops
        ops = self._apply_rule_homothety(ops)
        return NonCommutativeCompositeOperator._apply_rules(self, ops)

    def _apply_rule_homothety(self, operands):
        """
        Group scalars from homothety operators and try to inject the result
        into operators that can absorb scalars.

        """
        return sum((self._apply_rule_homothety_linear(list(group))
                    if linear else list(group) for linear, group in
                    groupby(operands, lambda o: o.flags.linear)), [])

    def _apply_rule_homothety_linear(self, operands):
        if len(operands) <= 1:
            return operands
        scalar = np.array(1, bool)
        for i, op in enumerate(operands):
            if isinstance(op, IdentityOperator) or \
               not isinstance(op, HomothetyOperator):
                continue
            scalar = scalar * op.data
            operands[i] = _copy_direct(op, IdentityOperator())

        if scalar == 1:
            return operands

        # can the factor be absorbed by one of the operators?
        h = HomothetyOperator(scalar)
        try:
            for iop, op in enumerate(operands):
                if isinstance(op, IdentityOperator):
                    continue
                if CompositionOperator not in op.rules:
                    continue
                for rule in op.rules[CompositionOperator]['left']:
                    if rule.subjects != ('.', HomothetyOperator):
                        continue
                    try:
                        new_op = rule(op, h)
                    except:
                        continue
                    if new_op is not None:
                        raise StopIteration()
                for rule in op.rules[CompositionOperator]['right']:
                    if rule.subjects != (HomothetyOperator, '.'):
                        continue
                    try:
                        new_op = rule(h, op)
                    except:
                        continue
                    if new_op is not None:
                        raise StopIteration()
        except StopIteration:
            operands[iop] = _copy_direct(op, new_op)
        else:
            operands.insert(0, h)
        return operands

    def _get_info(self, input, output, preserve_input):
        """
        Given the context in which the composition is taking place:
            1) input and output shape, dtype, alignment and contiguity
            2) in-place or out-of-place composition
            3) whether the input should be preserved,

        the routine returns the requirements for the intermediate buffers of
        the composition and the information to perform the composition:
            1) output shape and dtype of each operator
            2) groups of operators that will operate on the same output buffer
        Except for the innermost group, which only contains in-place operators
        a group is an out-of-place operator followed by a certain number of
        in-place operators
            3) minimum buffer size, alignment and contiguity requirements
        for each group.

        For example, in the composition of I*I*O*I*O*I*I*I*O*I (I:in-place,
        O:out-of-place operator), the groups are 'IIO', 'IO', 'IIIO' and 'I'.
        For 'I*O', the groups are 'IO' and an empty group ''.

        """
        shapein = input.shape
        shapeout = output.shape
        dtypein = input.dtype
        dtypeout = output.dtype
        alignedin = input.__array_interface__['data'][0] \
            % config.MEMORY_ALIGNMENT == 0
        alignedout = output.__array_interface__['data'][0] \
            % config.MEMORY_ALIGNMENT == 0
        contiguousin = input.flags.contiguous
        contiguousout = output.flags.contiguous

        id_ = (shapein, shapeout, dtypein, dtypeout, alignedin, alignedout,
               contiguousin, contiguousout, preserve_input)

        try:
            return self._info[id_]
        except KeyError:
            pass

        shapes = self._get_shapes(shapein, shapeout, self.operands)[:-1]
        if None in shapes:
            raise ValueError(
                "The composition of an unconstrained input shape operator by a"
                "n unconstrained output shape operator is ambiguous.")
        dtypes = self._get_dtypes(input.dtype)
        sizes = [product(s) * d.itemsize for s, d in zip(shapes, dtypes)]

        ninplaces, aligneds, contiguouss = self._get_requirements()

        # make last operand out-of-place
        if preserve_input and self.operands[-1].flags.inplace or \
           not alignedin and aligneds[-1] or \
           not contiguousin and contiguouss[-1]:
            assert ninplaces[-1] > 0
            ninplaces[-1] -= 1
            ninplaces += [0]
            aligneds += [alignedin]
            contiguouss += [contiguousin]

        # make first operand out-of-place
        if sizes[0] < max([s for s in sizes[:ninplaces[0]+1]]) or \
           not alignedout and aligneds[0] or \
           not contiguousout and contiguouss[0]:
            assert ninplaces[0] > 0
            ninplaces[0] -= 1
            ninplaces.insert(0, 0)
            aligneds.insert(0, alignedout)
            contiguouss.insert(0, contiguousout)

        bufsizes = self._get_bufsizes(sizes, ninplaces)

        v = shapes, dtypes, ninplaces, bufsizes, aligneds, contiguouss
        self._info[id_] = v

        return v

    def _get_bufsizes(self, sizes, ninplaces):
        bufsizes = []
        iop = 0
        for n in ninplaces[:-1]:
            bufsizes.append(max(sizes[iop:iop+n+1]))
            iop += n + 1
        bufsizes.append(sizes[-1])
        return bufsizes

    def _get_dtypes(self, dtype):
        dtypes = []
        for op in self.operands[::-1]:
            dtype = self._find_common_type([dtype, op.dtype])
            dtypes.insert(0, dtype)
        return dtypes

    def _get_requirements(self):
        aligneds = []
        contiguouss = []
        ninplaces = []
        ninplace = 0
        aligned = False
        contiguity = False
        iop = len(self.operands) - 1

        # loop over operators
        while iop >= 0:

            # loop over in-place operators
            while iop >= 0:
                op = self.operands[iop]
                iop -= 1
                if not op.flags.inplace:
                    aligned = max(aligned, op.flags.aligned_input)
                    contiguity = max(contiguity, op.flags.contiguous_input)
                    break
                ninplace += 1
                aligned = max(aligned, op.flags.aligned_input)
                contiguity = max(contiguity, op.flags.contiguous_input)

            ninplaces.insert(0, ninplace)
            aligneds.insert(0, aligned)
            contiguouss.insert(0, contiguity)

            ninplace = 0
            aligned = op.flags.aligned_output
            contiguity = op.flags.contiguous_output

        if not op.flags.inplace:
            ninplaces.insert(0, ninplace)
            aligneds.insert(0, aligned)
            contiguouss.insert(0, contiguity)

        return ninplaces, aligneds, contiguouss

    @staticmethod
    def _get_shapes(shapein, shapeout, operands):
        """
        Return the output, intermediate and input shapes of the composed
        operands as a list.
        """
        n = len(operands)
        shapes = [shapeout] + (n - 1) * [None] + [shapein]

        # scanning from the innermost to the outermost operand
        for i in range(n-1, -1, -1):
            op = operands[i]
            if shapes[i+1] is None:
                s = op.shapeout
            else:
                s = tointtuple(op.reshapein(shapes[i+1]))
            if i == 0 and None not in (shapes[0], s) and s != shapes[0]:
                raise ValueError("Incompatible shape in composition.")
            if s is not None:
                shapes[i] = s

        # scanning from the outermost to the innermost operand
        for i in range(n):
            op = operands[i]
            if shapes[i] is None:
                s = op.shapein
            else:
                s = tointtuple(op.reshapeout(shapes[i]))
            if None not in (shapes[i+1], s) and s != shapes[i+1]:
                raise ValueError("Incompatible shape in composition.")
            if s is not None:
                shapes[i+1] = s

        return shapes

    @classmethod
    def _get_attributes(cls, operands, **keywords):
        shapes = cls._get_shapes(operands[-1].shapein, operands[0].shapeout,
                                 operands)
        attr = {
            'attrin': cls._merge_attr([o.attrin for o in operands]),
            'attrout': cls._merge_attr([o.attrout for o in operands[::-1]]),
            'classin': first_is_not((o.classin for o in operands[::-1]), None),
            'classout': first_is_not((o.classout for o in operands), None),
            'commin': first_is_not((o.commin for o in operands[::-1]), None),
            'commout': first_is_not((o.commout for o in operands), None),
            'dtype': cls._find_common_type(o.dtype for o in operands),
            'flags': cls._merge_flags(operands),
            'reshapein': cls._merge_reshapein(operands),
            'reshapeout': cls._merge_reshapeout(operands),
            'shapein': shapes[-1],
            'shapeout': shapes[0],
            'toshapein': operands[-1].toshapein,
            'toshapeout': operands[0].toshapeout,
            'validatein': operands[-1].validatein,
            'validateout': operands[0].validateout,
        }
        attr.update(keywords)
        return attr

    @classmethod
    def _merge(cls, op, op1, op2):
        """
        Ensure that op = op1*op2 has a correct shapein, shapeout, etc.

        """
        # bail if the merging has already been done
        if any(isinstance(o, CompositionOperator) for o in [op1, op2]):
            return
        keywords = cls._get_attributes([op1, op2], flags=op.flags)
        op._reset(**keywords)

    @staticmethod
    def _merge_attr(attrs):
        if all(a is None for a in attrs):
            return None
        if all(a is None or isinstance(a, dict) for a in attrs):
            attr = {}
            for a in attrs:
                if a is not None:
                    attr.update(a)
            return attr

        def func(attr):
            for a in attrs:
                if isinstance(a, dict):
                    attr.update(a)
                else:
                    a(attr)
        return func

    @staticmethod
    def _merge_flags(operands):
        return {
            'linear': all(op.flags.linear for op in operands),
            'real': all(op.flags.real for op in operands),
            'square': all(op.flags.square for op in operands),
            'separable': all(op.flags.separable for op in operands),
            'update_output': operands[0].flags.update_output,
            'aligned_input': operands[-1].flags.aligned_input,
            'aligned_output': operands[0].flags.aligned_output,
            'contiguous_input': operands[-1].flags.contiguous_input,
            'contiguous_output': operands[0].flags.contiguous_output}

    @staticmethod
    def _merge_reshapein(operands):
        if any(o.flags.shape_output != 'implicit' for o in operands):
            return None
        if all(o.flags.square for o in operands):
            return operands[-1].reshapein

        def reshapein(shape):
            for o in operands[::-1]:
                shape = tointtuple(o.reshapein(shape))
            return shape
        return reshapein

    @staticmethod
    def _merge_reshapeout(operands):
        if any(o.flags.shape_input != 'implicit' for o in operands):
            return None
        if all(o.flags.square for o in operands):
            return operands[0].reshapeout

        def reshapeout(shape):
            for o in operands:
                shape = tointtuple(o.reshapeout(shape))
            return shape
        return reshapeout

    def _validate_comm(self, operands):
        for op1, op2 in zip(operands[:-1], operands[1:]):
            commin = op1.commin
            commout = op2.commout
            if None not in (commin, commout) and commin is not commout:
                raise ValueError('The MPI communicators are incompatible.')
        return operands

    def __str__(self):
        if len(self.operands) == 0:
            return str(self.operands[0])

        s = ''
        for linear, group in groupby(reversed(self.operands),
                                     lambda _: _.flags.linear):
            group = tuple(group)
            if linear:
                s_group = ' * '.join(str(_) for _ in reversed(group))
                if len(s) == 0:
                    s = s_group
                    continue
                need_protection = len(group) > 1 or ' ' in s_group
                if need_protection:
                    s = '({0})({1})'.format(s_group, s)
                else:
                    s = s_group + '({0})'.format(s)
                continue
            for op in group:
                s_op = str(op)
                if len(s) == 0:
                    s = s_op
                    continue
                if '...' not in s_op:
                    s = '{0}({1})'.format(s_op, s)
                    continue
                protected = '...,' in s_op or ', ...' in s_op
                need_protection = ' ' in s #XXX fail for f(..., z=1)
                if not protected and need_protection:
                    s = s_op.replace('...', '({0})'.format(s))
                else:
                    s = s_op.replace('...', s)
        return s


class GroupOperator(CompositionOperator):
    """
    CompositionOperator subclass, without the associativity rules.

    Use this operator to make sure that properties such as dtype are not
    lost by composing with other operators.

    """
    def __init__(self, operands, **keywords):
        CompositionOperator.__init__(self, operands, **keywords)
        if not isinstance(self, GroupOperator):
            return

        dtype = self._find_common_type(o.dtype for o in self.operands)
        switch_T_H = self.flags.real and dtype is not None and \
            dtype.kind == 'c'
        if switch_T_H:
            T, H, IT, IH = 'H', 'T', 'IH', 'IT'
        else:
            T, H, IT, IH = 'T', 'H', 'IT', 'IH'

        self.set_rule('C', lambda s: GroupOperator(
            [m.C for m in s.operands], name=self.__name__ + '.C'))
        self.set_rule(T, lambda s: GroupOperator(
            [m.T for m in s.operands[::-1]], name=self.__name__ + '.T'))
        self.set_rule(H, lambda s: GroupOperator(
            [m.H for m in s.operands[::-1]], name=self.__name__ + '.H'))
        self.set_rule('I', lambda s: GroupOperator(
            [m.I for m in s.operands[::-1]], name=self.__name__ + '.I'))
        self.set_rule('IC', lambda s: GroupOperator(
            [m.I.C for m in s.operands[::-1]], name=self.__name__ + '.I.C'))
        self.set_rule(IT, lambda s: GroupOperator(
            [m.I.T for m in s.operands], name=self.__name__ + '.I.T'))
        self.set_rule(IH, lambda s: GroupOperator(
            [m.I.H for m in s.operands], name=self.__name__ + '.I.H'))
        self.del_rule(('.', CompositionOperator), CompositionOperator)
        self.del_rule(('.', Operator), CompositionOperator)
        self.del_rule((Operator, '.'), CompositionOperator)

    morph_single_operand = False


class BlockOperator(NonCommutativeCompositeOperator):
    """
    Abstract base class for BlockColumnOperator, BlockDiagonalOperator and
    BlockRowOperator.

    """
    def __init__(self, operands, partitionin=None, partitionout=None,
                 axisin=None, axisout=None, new_axisin=None, new_axisout=None,
                 **keywords):

        operands = self._validate_operands(operands)
        if len(operands) == 1:
            operands[0].copy(self)
            self._reset(**keywords)
            return

        if not isinstance(self, BlockRowOperator) and axisout is None and \
           new_axisout is None:
            self.__class__ = BlockRowOperator
            self.__init__(operands, partitionin, axisin, new_axisin)
            return
        if not isinstance(self, BlockColumnOperator) and axisin is None and \
           new_axisin is None:
            self.__class__ = BlockColumnOperator
            self.__init__(operands, partitionout, axisout, new_axisout)
            return
        if type(self) is BlockOperator:
            self.__class__ = BlockDiagonalOperator
            self.__init__(operands, partitionin, axisin, axisout, new_axisin,
                          new_axisout)
            return

        # from now on, self is of type Block(Column|Diagonal|Row)Operator
        if new_axisin is not None:
            if partitionin is None:
                partitionin = len(operands) * (1,)
            elif any(p not in (None, 1) for p in partitionin):
                raise ValueError(
                    'If the block operator input shape has one more dimension '
                    'than its blocks, the input partition must be a tuple of o'
                    'nes.')
        if new_axisout is not None:
            if partitionout is None:
                partitionout = len(operands) * (1,)
            elif any(p not in (None, 1) for p in partitionout):
                raise ValueError(
                    'If the block operator output shape has one more dimension'
                    ' than its blocks, the output partition must be a tuple of'
                    ' ones.')

        if axisin is not None and new_axisin is not None:
            raise ValueError("The keywords 'axisin' and 'new_axisin' are exclu"
                             "sive.")
        if axisout is not None and new_axisout is not None:
            raise ValueError("The keywords 'axisout' and 'new_axisout' are exc"
                             "lusive.")

        if partitionin is partitionout is None:
            raise ValueError('No partition is provided.')
        if partitionin is not None:
            if len(partitionin) != len(operands):
                raise ValueError('The number of operators must be the same as '
                                 'the length of the input partition.')
            partitionin = merge_none(partitionin, self._get_partitionin(
                operands, partitionout, axisin, axisout, new_axisin,
                new_axisout))
        if partitionout is not None:
            if len(partitionout) != len(operands):
                raise ValueError('The number of operators must be the same as '
                                 'the length of the output partition.')
            partitionout = merge_none(partitionout, self._get_partitionout(
                                      operands, partitionin, axisin, axisout,
                                      new_axisin, new_axisout))

        self.partitionin = tointtuple(partitionin)
        self.partitionout = tointtuple(partitionout)
        self.axisin = axisin
        self.new_axisin = new_axisin
        self.axisout = axisout
        self.new_axisout = new_axisout

        keywords = self._get_attributes(operands, **keywords)
        CompositeOperator.__init__(self, operands, **keywords)

        if self.shapein is not None:
            n = len(self.shapein)
            if self.axisin is not None and self.axisin < 0:
                self.axisin += n
            elif self.new_axisin is not None and self.new_axisin < 0:
                self.new_axisin += n
        if self.shapeout is not None:
            n = len(self.shapeout)
            if self.axisout is not None and self.axisout < 0:
                self.axisout += n
            elif self.new_axisout is not None and self.new_axisout < 0:
                self.new_axisout += n

        self.set_rule('C', lambda s: BlockOperator(
            [op.C for op in s.operands], s.partitionin, s.partitionout,
            s.axisin, s.axisout, s.new_axisin, s.new_axisout))
        self.set_rule('T', lambda s: BlockOperator(
            [op.T for op in s.operands], s.partitionout, s.partitionin,
            s.axisout, s.axisin, s.new_axisout, s.new_axisin))
        self.set_rule('H', lambda s: BlockOperator(
            [op.H for op in s.operands], s.partitionout, s.partitionin,
            s.axisout, s.axisin, s.new_axisout, s.new_axisin))

        if isinstance(self, BlockDiagonalOperator):
            self.set_rule('I', lambda s: type(s)(
                [op.I for op in s.operands], s.partitionout, s.axisout,
                s.axisin, s.new_axisout, s.new_axisin))
            self.set_rule('IC', lambda s: type(s)(
                [op.I.C for op in s.operands], s.partitionout, s.axisout,
                s.axisin, s.new_axisout, s.new_axisin))
            self.set_rule('IT', lambda s: type(s)(
                [op.I.T for op in s.operands], s.partitionin, s.axisin,
                s.axisout, s.new_axisin, s.new_axisout))
            self.set_rule('IH', lambda s: type(s)(
                [o.I.H for o in s.operands], s.partitionin, s.axisin,
                s.axisout, s.new_axisin, s.new_axisout))

        self.set_rule(('.', Operator), self._rule_operator_add,
                      AdditionOperator)
        self.set_rule(('.', Operator), self._rule_operator_mul,
                      MultiplicationOperator)
        self.set_rule(('.', Operator), self._rule_operator_rcomp,
                      CompositionOperator)
        self.set_rule((Operator, '.'), self._rule_operator_lcomp,
                      CompositionOperator)
        self.set_rule(('.', type(self)), self._rule_blocksameoperator_add,
                      AdditionOperator)
        self.set_rule(('.', type(self)), self._rule_blocksameoperator_mul,
                      MultiplicationOperator)
        self.set_rule(('.', BlockOperator), self._rule_blockoperator_comp,
                      CompositionOperator)

    def __mul__(self, other):
        if isinstance(other, BlockOperator) and not other.flags.linear:
            if isinstance(self, BlockRowOperator) and \
               isinstance(other, BlockDiagonalOperator) or \
               isinstance(self, BlockDiagonalOperator) and \
               isinstance(other, BlockColumnOperator) or \
               isinstance(self, BlockRowOperator) and \
               isinstance(other, BlockColumnOperator):
                new_op = self._rule_blockoperator_noncommutative(
                    self, other, MultiplicationOperator)
                if new_op is not None:
                    return new_op
        return NonCommutativeCompositeOperator.__mul__(self, other)

    def toshapein(self, v):
        if self.shapein is not None:
            return v.reshape(self.shapein)
        if self.partitionin is None:
            return self.operands[0].toshapein(v)
        axisin = self.axisin if self.axisin is not None else self.new_axisin
        if None in self.partitionin or axisin not in (0, -1):
            raise ValueError('Ambiguous reshaping.')
        p = sum(self.partitionin)
        if v.size == p:
            return v
        if axisin == 0:
            return v.reshape((p, -1))
        return v.reshape((-1, p))

    def toshapeout(self, v):
        if self.shapeout is not None:
            return v.reshape(self.shapeout)
        if self.partitionout is None:
            return self.operands[0].toshapeout(v)
        axisout = self.axisout if self.axisout is not None else \
            self.new_axisout
        if None in self.partitionout or axisout not in (0, -1):
            raise ValueError('Ambiguous reshaping.')
        p = sum(self.partitionout)
        if v.size == p:
            return v
        if axisout == 0:
            return v.reshape((p, -1))
        return v.reshape((-1, p))

    def _get_attributes(self, operands, **keywords):
        # UGLY HACK: required by self.reshapein/out. It may be better to make
        # the _get_attributes a class method, pass all partitionin/out etc
        # stuff and inline the reshapein/out methods to get shapein/shapeout.
        self.operands = operands

        attr = {
            'attrin': first_is_not((o.attrin for o in operands), None),
            'attrout': first_is_not((o.attrout for o in operands), None),
            'classin': first_is_not((o.classin for o in operands), None),
            'classout': first_is_not((o.classout for o in operands), None),
            'commin': first_is_not((o.commin for o in operands), None),
            'commout': first_is_not((o.commout for o in operands), None),
            'dtype': self._find_common_type(o.dtype for o in operands),
            'flags': self._merge_flags(operands),
            'shapein': self.reshapeout(None),
            'shapeout': self.reshapein(None),
        }
        for k, v in keywords.items():
            if k is not 'flags':
                attr[k] = v
        attr['flags'].update(
            Operator.validate_flags(keywords.get('flags', {})))
        return attr

    @staticmethod
    def _get_partition(shapes, axis, new_axis):
        if new_axis is not None:
            return len(shapes) * (1,)
        return tuple(None if s is None else s[axis] for s in shapes)

    @staticmethod
    def _get_partitionin(ops, partitionout, axisin, axisout, new_axisin,
                         new_axisout):
        """ Infer the input partition from the output partition. """
        if new_axisin is not None:
            return len(ops) * (1,)

        if partitionout is None:
            return [o.shapein[axisin] if o.shapein else None for o in ops]

        if new_axisout is None:
            ndim_min = axisout+1 if axisout >= 0 else -axisout
        else:
            ndim_min = 0
        partitionin = len(ops) * [None]
        for i, op in enumerate(ops):
            if op.shapein is not None:
                partitionin[i] = op.shapein[axisin]
                continue
            if partitionout[i] is None:
                continue
            pin = []
            # for implicit input shape operators, we should make sure that
            # partitionin does not depend on the rank of the output
            for ndim in range(ndim_min, 33):
                shapeout = ndim * [0]
                if new_axisout is None:
                    shapeout[axisout] = partitionout[i]
                try:
                    shapein = tointtuple(op.reshapeout(tuple(shapeout)))
                    pin.append(shapein[axisin])
                except IndexError:
                    continue
            if len(pin) == 0 or any(p != pin[0] for p in pin):
                continue
            partitionin[i] = pin[0]
        return tuple(partitionin)

    @staticmethod
    def _get_partitionout(ops, partitionin, axisin, axisout, new_axisin,
                          new_axisout):
        """ Infer the output partition from the input partition. """
        if new_axisout is not None:
            return len(ops) * (1,)

        if partitionin is None:
            return [o.shapeout[axisout] if o.shapeout else None for o in ops]

        if new_axisin is None:
            ndim_min = axisin+1 if axisin >= 0 else -axisin
        else:
            ndim_min = 0
        partitionout = len(ops) * [None]
        for i, op in enumerate(ops):
            if op.shapeout is not None:
                partitionout[i] = op.shapeout[axisout]
                continue
            if partitionin[i] is None:
                continue
            pout = []
            # for implicit output shape operators, we should make sure that
            # partitionout does not depend on the rank of the input
            for ndim in range(ndim_min, 33):
                shapein = ndim * [0]
                if new_axisin is None:
                    shapein[axisin] = partitionin[i]
                try:
                    shapeout = tointtuple(op.reshapein(tuple(shapein)))
                    pout.append(shapeout[axisout])
                except IndexError:
                    continue
            if len(pout) == 0 or any(p != pout[0] for p in pout):
                continue
            partitionout[i] = pout[0]
        return tuple(partitionout)

    @staticmethod
    def _get_shape_composite(shapes, p, axis, new_axis):
        """ Return composite shape from operand shapes. """
        explicit = [s for s in shapes if s is not None]
        if len(explicit) == 0:
            return None
        shape = explicit[0]

        if p is None or new_axis is not None:
            if any(s != shape for s in explicit):
                raise ValueError("The operands have incompatible shapes: '{0}'"
                                 ".".format(shapes))
            if p is None:
                return shape
            a = new_axis
            if new_axis < 0:
                a += len(shape) + 1
            return shape[:a] + (len(p),) + shape[a:]

        rank = len(shape)
        if any(len(s) != rank for s in explicit):
            raise ValueError(
                "The blocks do not have the same number of dimensions: '{0}'.".
                format(shapes))
        if any(shapes[i] is not None and shapes[i][axis] != p[i]
                for i in range(len(p)) if p[i] is not None):
            raise ValueError(
                "The blocks have shapes '{0}' incompatible with the partition "
                "{1}.".format(shapes, p))
        if len(explicit) != 1:
            ok = [all(s is None or s[i] == shape[i] for s in shapes)
                  for i in range(rank)]
            ok[axis] = True
            if not all(ok):
                raise ValueError(
                    "The dimensions of the blocks '{0}' are not the same along"
                    " axes other than that of the partition '{1}'.".format(
                    shapes, p))

        p = merge_none(p, [s[axis] if s is not None else None for s in shapes])
        if None in p:
            return None

        shape = list(shape)
        shape[axis] = sum(p)
        return tointtuple(shape)

    @staticmethod
    def _get_shape_operands(shape, partition, partition_other, axis, new_axis):
        """ Return operand shapes from composite shape. """
        if partition is None:
            return len(partition_other) * (shape,)
        if None in partition or shape is None:
            return len(partition) * (None,)
        if new_axis is not None:
            shape_ = list(shape)
            del shape_[new_axis]
            shapes = len(partition) * (tuple(shape_),)
            return shapes
        shapes = []
        for p in partition:
            shape_ = list(shape)
            shape_[axis] = p
            shapes.append(tuple(shape_))
        return tuple(shapes)

    @staticmethod
    def _get_slices(partition, axis, new_axis):
        """ Return an iterator of the block slices. """
        if new_axis is not None:
            axis = new_axis
        if axis >= 0:
            s = (axis+1) * [slice(None)] + [Ellipsis]
        else:
            s = [Ellipsis] + (-axis) * [slice(None)]
        dest = 0
        for n in partition:
            if new_axis is not None:
                s[new_axis] = dest
            else:
                s[axis] = slice(dest, dest + n)
            dest += n
            yield list(s)

    def get_slicesin(self, partitionin=None):
        """ Return an iterator of the block input slices. """
        if partitionin is None:
            partitionin = self.partitionin
        return self._get_slices(partitionin, self.axisin, self.new_axisin)

    def get_slicesout(self, partitionout=None):
        """ Return an iterator of the block output slices. """
        if partitionout is None:
            partitionout = self.partitionout
        return self._get_slices(partitionout, self.axisout, self.new_axisout)

    @staticmethod
    def _merge_flags(operands):
        return {'linear': all(op.flags.linear for op in operands),
                'real': all(op.flags.real for op in operands)}

    def reshapein(self, shapein):
        shapeins = self._get_shape_operands(
            shapein, self.partitionin, self.partitionout, self.axisin,
            self.new_axisin)
        shapeouts = [o.shapeout if s is None else tointtuple(o.reshapein(s))
                     for o, s in zip(self.operands, shapeins)]
        return self._get_shape_composite(shapeouts, self.partitionout,
                                         self.axisout, self.new_axisout)

    def reshapeout(self, shapeout):
        shapeouts = self._get_shape_operands(
            shapeout, self.partitionout, self.partitionin, self.axisout,
            self.new_axisout)
        shapeins = [o.shapein if s is None else tointtuple(o.reshapeout(s))
                    for o, s in zip(self.operands, shapeouts)]
        return self._get_shape_composite(shapeins, self.partitionin,
                                         self.axisin, self.new_axisin)

    @staticmethod
    def _validate_partition_commutative(op1, op2):
        axisin1 = op1.axisin if op1.axisin is not None else op1.new_axisin
        axisin2 = op2.axisin if op2.axisin is not None else op2.new_axisin
        axisout1 = op1.axisout if op1.axisout is not None else op1.new_axisout
        axisout2 = op2.axisout if op2.axisout is not None else op2.new_axisout
        if axisin1 != axisin2 or axisout1 != axisout2:
            return None
        if op1.axisin is not None and op2.new_axisin is not None or \
           op1.new_axisin is not None and op2.axisin is not None or \
           op1.axisout is not None and op2.new_axisout is not None or \
           op1.new_axisout is not None and op2.axisout is not None:
            #XXX we could handle these cases with a reshape
            return None
        try:
            return merge_none(op1.partitionout, op2.partitionout), \
                   merge_none(op1.partitionin, op2.partitionin)
        except ValueError:
            return None

    @staticmethod
    def _validate_partition_composition(op1, op2):
        axisin1 = first_is_not([op1.axisin, op1.new_axisin], None)
        axisout2 = first_is_not([op2.axisout, op2.new_axisout], None)
        if axisin1 < 0 and op2.shapeout is not None:
            axisin1 += len(op2.shapeout)
        if axisout2 < 0 and op1.shapein is not None:
            axisout2 += len(op1.shapein)
        if axisin1 != axisout2:
            return None
        if op1.axisin is not None and op2.new_axisout is not None or \
           op1.new_axisin is not None and op2.axisout is not None:
            #XXX we could handle these cases with a reshape
            return None
        p1 = op1.partitionin
        p2 = op2.partitionout
        if p1 is None or p2 is None:
            return None
        try:
            p = merge_none(p1, p2)
        except ValueError:
            return None
        pout = None if op1.partitionout is None else op1._get_partitionout(
            op1.operands, p, op1.axisin, op1.axisout, op1.new_axisin,
            op1.new_axisout)
        pin = None if op2.partitionin is None else op2._get_partitionin(
            op2.operands, p, op2.axisin, op2.axisout, op2.new_axisin,
            op2.new_axisout)

        return None if pout is None else merge_none(op1.partitionout, pout), \
               None if pin is None else merge_none(op2.partitionin, pin)

    @staticmethod
    def _rule_operator_commutative(self, op, cls):
        if not op.flags.separable:
            return None
        return BlockOperator(
            [cls([o, op]) for o in self.operands], self.partitionin,
            self.partitionout, self.axisin, self.axisout, self.new_axisin,
            self.new_axisout)

    @staticmethod
    def _rule_operator_add(self, op):
        """ Rule for BlockOperator + Operator. """
        return self._rule_operator_commutative(self, op, AdditionOperator)

    @staticmethod
    def _rule_operator_mul(self, op):
        """ Rule for BlockOperator x Operator. """
        return self._rule_operator_commutative(self, op,
                                               MultiplicationOperator)

    @staticmethod
    def _rule_operator_lcomp(op, self):
        """ Rule for Operator(BlockOperator). """
        if self.partitionout is None:
            return None
        if isinstance(op, BlockOperator):
            return None
        if not op.flags.separable:
            return None
        n = len(self.partitionout)
        partitionout = self._get_partitionout(
            n * [op], self.partitionout, self.axisout, self.axisout,
            self.new_axisout, self.new_axisout)
        return BlockOperator(
            [op(o) for o in self.operands], self.partitionin, partitionout,
            self.axisin, self.axisout, self.new_axisin, self.new_axisout)

    @staticmethod
    def _rule_operator_rcomp(self, op):
        """ Rule for BlockOperator(Operator). """
        if self.partitionin is None:
            return None
        if not op.flags.separable:
            return None
        n = len(self.partitionin)
        partitionin = self._get_partitionin(
            n * [op], self.partitionin, self.axisin, self.axisin,
            self.new_axisin, self.new_axisin)
        return BlockOperator(
            [o(op) for o in self.operands], partitionin, self.partitionout,
            self.axisin, self.axisout, self.new_axisin, self.new_axisout)

    @staticmethod
    def _rule_blocksameoperator_commutative(p1, p2, operation):
        partitions = p1._validate_partition_commutative(p1, p2)
        if partitions is None:
            return None
        partitionout, partitionin = partitions
        operands = [operation([o1, o2]) for o1, o2 in
                    zip(p1.operands, p2.operands)]
        return BlockOperator(
            operands, partitionin, partitionout, p1.axisin, p1.axisout,
            p1.new_axisin, p1.new_axisout)

    @staticmethod
    def _rule_blocksameoperator_add(p1, p2):
        """ Rule for same type BlockOperator + BlockOperator. """
        return p1._rule_blocksameoperator_commutative(p1, p2, AdditionOperator)

    @staticmethod
    def _rule_blocksameoperator_mul(p1, p2):
        """ Rule for same type BlockOperator x BlockOperator. """
        return p1._rule_blocksameoperator_commutative(p1, p2,
                                                      MultiplicationOperator)

    @staticmethod
    def _rule_blockoperator_noncommutative(p1, p2, cls):
        partitions = p1._validate_partition_composition(p1, p2)
        if partitions is None:
            return None
        partitionout, partitionin = partitions
        operands = [cls([o1, o2]) for o1, o2 in zip(p1.operands, p2.operands)]
        if partitionin is partitionout is None:
            return AdditionOperator(operands)
        axisin, axisout = p2.axisin, p1.axisout
        new_axisin, new_axisout = p2.new_axisin, p1.new_axisout
        return BlockOperator(
            operands, partitionin, partitionout, axisin, axisout, new_axisin,
            new_axisout)

    @staticmethod
    def _rule_blockoperator_comp(p, q):
        """ Rule for BlockOperator(BlockOperator). """
        return p._rule_blockoperator_noncommutative(p, q, CompositionOperator)


class BlockDiagonalOperator(BlockOperator):
    """
    Block diagonal operator.

    If a new axis 'new_axisin' is specified, the input shapes of the blocks
    must be the same, and the input is iterated along this axis. Otherwise,
    the input shapes of the blocks must be the same except for one same
    dimension 'axisin': the axis along which the input is partitioned.

    If a new axis 'new_axisout' is specified, the output shapes of the blocks
    must be the same, and the output is stacked along this axis. Otherwise,
    the output shapes of the blocks must be the same except for one same
    dimension 'axisout': the axis along which the output is partitioned.
    This operator can be used to process data chunk by chunk.

    This operator can be used to process data chunk by chunk.

    The direct methods of the partition operators may be called with non-C or
    non-Fortran contiguous input and output arrays, so care must be taken when
    interfacing C or Fortran code.

    Parameters
    ----------
    operators : Operator list
        Operators that will populate the diagonal blocks.
    partitionin : tuple of int
        Partition of the number of elements along the input partition axis, to
        be provided if at least one of the input operators is implicit-shape
    axisin : int
        Input partition axis (default is 0)
    axisout : int
        Output partition axis (default is the input partition axis)

    Example
    -------
    o1, o2 = Operator(shapein=(16,4)), Operator(shapein=(16,3))
    p = BlockDiagonalOperator([o1, o2], axisin=-1)
    print(p.shapein)
    (16,7)

    """
    def __init__(self, operands, partitionin=None, axisin=None, axisout=None,
                 new_axisin=None, new_axisout=None, **keywords):

        operands = self._validate_operands(operands)

        if axisout is None:
            axisout = axisin
        if new_axisout is None:
            new_axisout = new_axisin
        if axisin is None:
            axisin = axisout
        if new_axisin is None:
            new_axisin = new_axisout

        if axisin is None and new_axisin is None:
            raise NotImplementedError('Free partitioning not implemented yet.')

        if partitionin is None:
            partitionin = self._get_partition(
                [op.shapein for op in operands], axisin, new_axisin)
        partitionin = tointtuple(partitionin)
        partitionout = len(partitionin) * (None,)

        BlockOperator.__init__(self, operands, partitionin, partitionout,
                               axisin, axisout, new_axisin, new_axisout,
                               **keywords)

    def direct(self, input, output):
        if None in self.partitionout:
            partitionout = list(self.partitionout)
            for i, o in enumerate(self.operands):
                if partitionout[i] is not None:
                    continue
                if self.partitionin[i] is None:
                    raise ValueError('The shape of an operator with implicit p'
                                     'artition cannot be inferred.')
                shapein = list(input.shape)
                shapein[self.axisin] = self.partitionin[i]
                partitionout[i] = tointtuple(
                    o.reshapein(shapein))[self.axisout]
        else:
            partitionout = self.partitionout

        for op, sin, sout in zip(self.operands, self.get_slicesin(),
                                 self.get_slicesout(partitionout)):
            i = input[sin]
            o = output[sout]
            with _pool.copy_if(i, op.flags.aligned_input,
                               op.flags.contiguous_input) as i:
                with _pool.copy_if(o, op.flags.aligned_output,
                                   op.flags.contiguous_output) as o:
                    op.direct(i, o)

    @staticmethod
    def _merge_flags(operands):
        flags = BlockOperator._merge_flags(operands)
        flags.update({'square': all(op.flags.square for op in operands),
                      'symmetric': all(op.flags.symmetric for op in operands),
                      'hermitian': all(op.flags.hermitian for op in operands),
                      'inplace': all(op.flags.inplace for op in operands)})
        return flags


class BlockColumnOperator(BlockOperator):
    """
    Block column operator.

    The input shapes of the blocks must be the same.
    If a new axis 'new_axisout' is specified, the output shapes of the blocks
    must be the same, and the output is stacked along this axis. Otherwise,
    the output shapes of the blocks must be the same except for one same
    dimension 'axisout': the axis along which the output is partitioned.
    This operator can be used to process data chunk by chunk.

    Example
    -------
    >>> I = IdentityOperator(shapein=3)
    >>> op = BlockColumnOperator([I,2*I], axisout=0)
    >>> op.todense()
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [2, 0, 0],
           [0, 2, 0],
           [0, 0, 2]])

    """
    def __init__(self, operands, partitionout=None, axisout=None,
                 new_axisout=None, **keywords):

        operands = self._validate_operands(operands)

        if axisout is None and new_axisout is None:
            raise NotImplementedError('Free partitioning not implemented yet.')

        if partitionout is None:
            partitionout = self._get_partition(
                [op.shapeout for op in operands], axisout, new_axisout)
        partitionout = tointtuple(partitionout)

        BlockOperator.__init__(self, operands, partitionout=partitionout,
                               axisout=axisout, new_axisout=new_axisout,
                               **keywords)

    def direct(self, input, output):
        if None in self.partitionout:
            partitionout = list(self.partitionout)
            for i, op in enumerate(self.operands):
                if partitionout[i] is None:
                    partitionout[i] = tointtuple(op.reshapein(input.shape)
                                                 [self.axisout])
        else:
            partitionout = self.partitionout

        for op, sout in zip(self.operands, self.get_slicesout(partitionout)):
            o = output[sout]
            with _pool.copy_if(o, op.flags.aligned_output,
                               op.flags.contiguous_output) as o:
                op.direct(input, o)

    def __str__(self):
        operands = ['[{0}]'.format(o) for o in self.operands]
        if len(operands) > 2:
            operands = [operands[0], '...', operands[-1]]
        return '[ ' + ' '.join(operands) + ' ]'


class BlockRowOperator(BlockOperator):
    """
    Block row operator.

    The output shapes of the blocks must be the same.
    If a new axis 'new_axisin' is specified, the input shapes of the blocks
    must be the same, and the input is iterated along this axis. Otherwise,
    the input shapes of the blocks must be the same except for one same
    dimension 'axisin': the axis along which the input is partitioned.
    This operator can be used to process data chunk by chunk.

    Example
    -------
    >>> I = IdentityOperator(shapein=3)
    >>> op = BlockRowOperator([I,2*I], axisin=0)
    >>> op.todense()
    array([[1, 0, 0, 2, 0, 0],
           [0, 1, 0, 0, 2, 0],
           [0, 0, 1, 0, 0, 2]])

    """
    def __init__(self, operands, partitionin=None, axisin=None,
                 new_axisin=None, operation=operator.iadd, **keywords):

        operands = self._validate_operands(operands)

        if axisin is None and new_axisin is None:
            raise NotImplementedError('Free partitioning not implemented yet.')

        if partitionin is None:
            partitionin = self._get_partition(
                [op.shapein for op in operands], axisin, new_axisin)
        partitionin = tointtuple(partitionin)

        keywords['flags'] = Operator.validate_flags(
            keywords.get('flags', {}), linear=operation is operator.iadd)
        BlockOperator.__init__(self, operands, partitionin=partitionin, axisin=
                               axisin, new_axisin=new_axisin, **keywords)

        self.operation = operation
        self._need_temporary = any(not o.flags.update_output for o in
                                   self.operands[1:])

    def direct(self, input, output):
        if None in self.partitionin:
            partitionin = list(self.partitionin)
            for i, op in enumerate(self.operands):
                if partitionin[i] is None:
                    partitionin[i] = tointtuple(op.reshapeout(output.shape)
                                                [self.axisin])
        else:
            partitionin = self.partitionin

        sins = tuple(self.get_slicesin(partitionin))
        i = input[sins[0]]
        op = self.operands[0]
        with _pool.copy_if(i, op.flags.aligned_input,
                           op.flags.contiguous_input) as i:
            op.direct(i, output)

        with _pool.get_if(self._need_temporary, output.shape, output.dtype,
                          self.__name__) as buf:

            for op, sin in zip(self.operands[1:], sins[1:]):
                i = input[sin]
                with _pool.copy_if(i, op.flags.aligned_input,
                                   op.flags.contiguous_input) as i:
                    if op.flags.update_output:
                        op.direct(i, output, operation=self.operation)
                    else:
                        op.direct(i, buf)
                        self.operation(output, buf)

    def __str__(self):
        operands = [str(o) for o in self.operands]
        if len(operands) > 2:
            operands = [operands[0], '...', operands[-1]]
        return '[[ ' + ' '.join(operands) + ' ]]'


@real
@linear
@inplace
class ReshapeOperator(Operator):
    """
    Operator that reshapes arrays.

    Example
    -------
    >>> op = ReshapeOperator(6, (3,2))
    >>> op(np.ones(6)).shape
    (3, 2)
    """
    def __init__(self, shapein, shapeout, **keywords):
        if shapein is None:
            raise ValueError('The input shape is None.')
        if shapeout is None:
            raise ValueError('The output shape is None.')
        shapein = tointtuple(shapein)
        shapeout = tointtuple(shapeout)
        if product(shapein) != product(shapeout):
            raise ValueError('The total size of the output must be unchanged.')
        if shapein == shapeout:
            self.__class__ = IdentityOperator
            self.__init__(shapein, **keywords)
            return
        Operator.__init__(self, shapein=shapein, shapeout=shapeout, **keywords)
        self.set_rule('T', lambda s: ReshapeOperator(s.shapeout, s.shapein))
        self.set_rule(('.', ReshapeOperator),
                      lambda s, o: ReshapeOperator(o.shapein, s.shapeout),
                      CompositionOperator)

    def direct(self, input, output):
        if isalias(input, output):
            pass
        output.ravel()[:] = input.ravel()

    def __str__(self):
        return strshape(self.shapeout) + '←' + strshape(self.shapein)


class BroadcastingBase(Operator):
    """
    Abstract class for operators that operate on a data array and
    the input array, and for which broadcasting of the data array across
    the input array is required.

    Leftward broadcasting is the normal numpy's broadcasting along the slow
    dimensions, if the array is stored in C order. Rightward broadcasting is
    a broadcasting along the fast dimensions.

    The following classes subclass BroadcastingBase :

    BroadcastingBase
        > ConstantOperator
        > DiagonalBase
              > DiagonalOperator
              > DiagonalNumexprOperator
              > MaskOperator
        > PackOperator
        > UnpackOperator

    """
    def __init__(self, data, broadcast, **keywords):
        if broadcast is None:
            raise ValueError('The broadcast mode is not specified.')
        data = np.asarray(data)
        broadcast = broadcast.lower()
        values = ('leftward', 'rightward', 'disabled', 'scalar')
        if broadcast not in values:
            raise ValueError(
                "Invalid value '{0}' for the broadcast keyword. Expected value"
                "s are {1}.".format(broadcast, strenum(values)))
        if data.ndim == 0 and broadcast in ('leftward', 'rightward'):
            broadcast = 'scalar'
        self.broadcast = broadcast
        self.data = data
        Operator.__init__(self, **keywords)
        self.set_rule(('.', BlockOperator),
                      lambda s, o: s._rule_right_block(
                          s, o, CompositionOperator), CompositionOperator)
        self.set_rule((BlockOperator, '.'),
                      lambda o, s: s._rule_left_block(o, s),
                      CompositionOperator)
        self.set_rule(('.', BlockOperator),
                      lambda s, o: s._rule_right_block(s, o, AdditionOperator),
                      AdditionOperator)
        self.set_rule(('.', BlockOperator),
                      lambda s, o: s._rule_right_block(
                          s, o, MultiplicationOperator),
                      MultiplicationOperator)

    @property
    def nbytes(self):
        return self.data.nbytes

    def get_data(self):
        return self.data

    @staticmethod
    def _rule_broadcast(b1, b2, cls, operation):
        b = set([b1.broadcast, b2.broadcast])
        if 'leftward' in b and 'rightward' in b:
            return None
        if 'disabled' in b:
            broadcast = 'disabled'
        elif 'leftward' in b:
            broadcast = 'leftward'
        elif 'rightward' in b:
            broadcast = 'rightward'
        else:
            broadcast = 'scalar'
        if 'rightward' in b:
            data = operation(b1.get_data().T, b2.get_data().T).T
        else:
            data = operation(b1.get_data(), b2.get_data())

        return cls(data, broadcast)

    @staticmethod
    def _rule_block(self, op, shape, partition, axis, new_axis, func_operation,
                    *args, **keywords):
        if partition is None:
            return
        if None in partition and self.broadcast != 'scalar':
            return

        b = self.broadcast
        ndim = self.data.ndim
        axis_ = first_is_not([axis, new_axis], None)

        # determine if the broadcasting data should be replicated or sliced
        do_replicate = False
        if b == 'scalar':
            if shape is None:
                do_replicate = True
        elif b == 'disabled':
            pass
        elif shape is None:
            if new_axis is not None and ndim == 1 and (
                    new_axis == -1 and b == 'rightward' or
                    new_axis == 0 and b == 'leftward'):
                do_replicate = True
            elif b == 'rightward':
                if axis_ > ndim:
                    do_replicate = True
                elif axis_ < 0:
                    return
            else:
                if axis_ < -ndim:
                    do_replicate = True
                elif axis_ >= 0:
                    return
        else:
            if b == 'rightward':
                if axis_ >= ndim:
                    do_replicate = True
            else:
                if axis is not None:
                    axis = axis - len(shape)
                else:
                    new_axis = new_axis - len(shape)
                if axis_ - len(shape) < -ndim:
                    do_replicate = True

        if do_replicate:
            ops = [func_operation(self, o) for o in op.operands]
        else:
            data = self._as_strided(shape)
            argspec = inspect.getargspec(type(self).__init__)
            nargs = len(argspec.args) - 1 - (
                len(argspec.defaults) if argspec.defaults is not None else 0)
            slices = op._get_slices(partition, axis, new_axis)
            ops = []
            for s, o in zip(slices, op.operands):
                if nargs == 0:
                    sliced = type(self)(*args, **keywords)
                else:
                    sliced = type(self)(data[s], broadcast=b,
                                        *args, **keywords)
                ops.append(func_operation(sliced, o))

        return BlockOperator(ops, op.partitionin, op.partitionout, op.axisin,
                             op.axisout, op.new_axisin, op.new_axisout)

    @staticmethod
    def _rule_left_block(op, self):
        func_op = lambda o, b: CompositionOperator([b, o])
        return self._rule_block(self, op, op.shapein, op.partitionin,
                                op.axisin, op.new_axisin, func_op)

    @staticmethod
    def _rule_right_block(self, op, cls):
        func_op = lambda o, b: cls([o, b])
        return self._rule_block(self, op, op.shapeout, op.partitionout,
                                op.axisout, op.new_axisout, func_op)

    def _as_strided(self, shape):
        if shape is None:
            return self.data
        strides = len(shape) * [0]
        if self.broadcast == 'rightward':
            delta = 0
        else:
            delta = len(shape) - self.data.ndim
        v = self.data.itemsize
        for i in range(self.data.ndim-1, -1, -1):
            s = self.data.shape[i]
            if s == 1:
                continue
            strides[i+delta] = v
            v *= s
        return np.lib.stride_tricks.as_strided(self.data, shape, strides)


@square
@symmetric
class DiagonalBase(BroadcastingBase):
    """
    Base class for DiagonalOperator, DiagonalNumexprOperator, MaskOperator.

    """
    def __init__(self, data, broadcast, **keywords):
        BroadcastingBase.__init__(self, data, broadcast, **keywords)
        self.set_rule(('.', DiagonalBase),
                      lambda s, o: s._rule_broadcast(
                          s, o, DiagonalOperator, np.add),
                      AdditionOperator)
        self.set_rule(('.', ConstantOperator),
                      lambda s, o: s._rule_broadcast(
                          s, o, DiagonalOperator, np.multiply),
                      MultiplicationOperator)
        self.set_rule(('.', DiagonalBase),
                      lambda s, o: s._rule_multiply(s, o),
                      MultiplicationOperator)
        self.set_rule(('.', DiagonalBase),
                      lambda s, o: s._rule_broadcast(
                          s, o, DiagonalOperator, np.multiply),
                      CompositionOperator)

    @staticmethod
    def _rule_multiply(b1, b2):
        b = set([b1.broadcast, b2.broadcast])
        if 'leftward' in b and 'rightward' in b:
            return
        if 'disabled' in b:
            b = 'disabled'
        elif 'leftward' in b:
            b = 'leftward'
        elif 'rightward' in b:
            b = 'rightward'
        else:
            b = 'scalar'
        if 'rightward' in b:
            data = (b1.get_data().T * b2.get_data().T).T
        else:
            data = b1.get_data() * b2.get_data()
        return MultiplicationOperator(
            [ConstantOperator(data, broadcast=b),
             po.nonlinear.PowerOperator(2)])


@inplace
class DiagonalOperator(DiagonalBase):
    """
    Diagonal operator.

    Arguments
    ---------
    data : ndarray
      The diagonal coefficients

    broadcast : 'rightward' or 'disabled' (default 'disabled')
      If broadcast == 'rightward', the diagonal is broadcasted along the fast
      axis.

    Exemple
    -------
    >>> A = DiagonalOperator(np.arange(1, 6, 2))
    >>> A.todense()
    array([[1, 0, 0],
           [0, 3, 0],
           [0, 0, 5]])

    >>> A = DiagonalOperator([1, 2], broadcast='rightward', shapein=(2, 2))
    >>> A.todense()
    array([[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 2, 0],
           [0, 0, 0, 2]])

    """
    def __init__(self, data, broadcast=None, dtype=None, **keywords):
        data = np.asarray(data)
        if broadcast is None:
            broadcast = 'scalar' if data.ndim == 0 else 'disabled'
        if broadcast == 'disabled':
            keywords['shapein'] = data.shape
            keywords['shapeout'] = data.shape
        n = data.size
        nmones, nzeros, nones, other, same = inspect_special_values(data)
        if nzeros == n and not isinstance(self, ZeroOperator):
            keywords['flags'] = Operator.validate_flags(
                keywords.get('flags', {}), square=True)
            self.__class__ = ZeroOperator
            self.__init__(dtype=dtype, **keywords)
            return
        if nones == n and not isinstance(self, IdentityOperator):
            self.__class__ = IdentityOperator
            self.__init__(dtype=dtype, **keywords)
            return
        if same and not isinstance(self, (HomothetyOperator, ZeroOperator)):
            self.__class__ = HomothetyOperator
            self.__init__(data.flat[0], dtype=dtype, **keywords)
            return
        if nones + nzeros == n and not isinstance(self,
                                                  (HomothetyOperator,
                                                   po.linear.MaskOperator)):
            self.__class__ = po.linear.MaskOperator
            self.__init__(~data.astype(np.bool8), broadcast=broadcast,
                          **keywords)
            return
        if nmones + nones == n:
            keywords['flags'] = self.validate_flags(keywords.get('flags', {}),
                                                    involutary=True)
        if dtype is None and (data.ndim > 0 or data not in (0, 1)):
            dtype = data.dtype
        DiagonalBase.__init__(self, data, broadcast, dtype=dtype, **keywords)

    def direct(self, input, output):
        if self.broadcast == 'rightward':
            np.multiply(input.T, self.get_data().T, output.T)
        else:
            np.multiply(input, self.get_data(), output)

    def conjugate(self, input, output):
        if self.broadcast == 'rightward':
            np.multiply(input.T, np.conjugate(self.get_data()).T, output.T)
        else:
            np.multiply(input, np.conjugate(self.get_data()), output)

    def inverse(self, input, output):
        if self.broadcast == 'rightward':
            np.divide(input.T, self.get_data().T, output.T)
        else:
            np.divide(input, self.get_data(), output)

    def inverse_conjugate(self, input, output):
        if self.broadcast == 'rightward':
            np.divide(input.T, np.conjugate(self.get_data()).T, output.T)
        else:
            np.divide(input, np.conjugate(self.get_data()), output)

    def __pow__(self, n):
        if n in (-1, 0, 1):
            return BroadcastingBase.__pow__(self, n)
        return DiagonalOperator(self.get_data()**n, broadcast=self.broadcast)

    def validatein(self, shape):
        n = self.data.ndim
        if len(shape) < n:
            raise ValueError("Invalid number of dimensions.")

        if self.broadcast == 'rightward':
            it = zip(shape[:n], self.data.shape[:n])
        else:
            it = zip(shape[-n:], self.data.shape[-n:])
        for si, sd in it:
            if sd != 1 and sd != si:
                raise ValueError("The data array cannot be broadcast across th"
                                 "e input.")

    def toshapein(self, v):
        if self.shapein is not None:
            return v.reshape(self.shapein)
        if self.data.ndim < 1:
            return v

        sd = list(self.data.shape)
        n = sd.count(1)
        if n > 1:
            raise ValueError('Ambiguous broadcasting.')
        if n == 0:
            if self.broadcast == 'rightward':
                sd.append(-1)
            else:
                sd.insert(0, -1)
        else:
            sd[sd.index(1)] = -1

        try:
            v = v.reshape(sd)
        except ValueError:
            raise ValueError("Invalid broadcasting.")

        return v


class HomothetyOperator(DiagonalOperator):
    """
    Multiplication by a scalar.

    """
    def __init__(self, data, **keywords):
        data = np.asarray(data)
        if data.ndim > 0:
            if any(s != 0 for s in data.strides) and \
               np.any(data.flat[0] != data):
                raise ValueError("The input is not a scalar.")
            data = np.asarray(data.flat[0])

        DiagonalOperator.__init__(self, data, **keywords)
        if type(self) is not HomothetyOperator:
            return
        self.set_rule('C', lambda s: HomothetyOperator(np.conjugate(s.data)))
        self.set_rule('I', lambda s: HomothetyOperator(
            1/s.data if s.data != 0 else np.nan))
        self.set_rule('IC', lambda s: HomothetyOperator(
            np.conjugate(1/s.data) if s.data != 0 else np.nan))

    def __str__(self):
        data = self.data.flat[0]
        if data == int(data):
            data = int(data)
        if data == 1:
            return 'I'
        if data == -1:
            return '-I'
        return str(data) + 'I'


@real
@idempotent
@involutary
class IdentityOperator(HomothetyOperator):
    """
    A subclass of HomothetyOperator with data = 1.

    Examples
    --------
    >>> I = IdentityOperator()
    >>> I.todense(shapein=3)
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]])

    >>> I = IdentityOperator(shapein=2)
    >>> I * np.arange(2)
    array([0, 1])

    """
    def __init__(self, shapein=None, **keywords):
        HomothetyOperator.__init__(self, 1, shapein=shapein, **keywords)
        self.set_rule(('.', Operator), self._rule_left, CompositionOperator)
        self.set_rule((Operator, '.'), self._rule_right, CompositionOperator)

    def direct(self, input, output):
        if isalias(input, output):
            pass
        output[...] = input

    @staticmethod
    def _rule_left(self, operator):
        return operator.copy()

    @staticmethod
    def _rule_right(operator, self):
        return operator.copy()


@idempotent
@inplace
@update_output
class ConstantOperator(BroadcastingBase):
    """
    Non-linear constant operator.

    """
    def __init__(self, data, broadcast=None, dtype=None, **keywords):
        data = np.asarray(data)
        if broadcast is None:
            broadcast = 'scalar' if data.ndim == 0 else 'disabled'
        if broadcast == 'disabled':
            keywords['shapeout'] = data.shape
        if data.ndim > 0 and np.all(data == data.flat[0]):
            self.__init__(data.flat[0], dtype=dtype, **keywords)
            return
        if not isinstance(self, ZeroOperator) and data.ndim == 0 and data == 0:
            self.__class__ = ZeroOperator
            self.__init__(dtype=dtype, **keywords)
            return

        if dtype is None and not isinstance(self, ZeroOperator):
            dtype = data.dtype
        BroadcastingBase.__init__(self, data, broadcast, dtype=dtype,
                                  **keywords)
        self.set_rule('C', lambda s: ConstantOperator(
            s.data.conjugate(), broadcast=s.broadcast))
#        if self.flags.shape_input == 'unconstrained' and \
#           self.flags.shape_output != 'implicit':
#            self.set_rule('T', '.')
        self.set_rule(('.', Operator), self._rule_left, CompositionOperator)
        self.set_rule((Operator, '.'), self._rule_right, CompositionOperator)
        self.set_rule(('.', Operator), self._rule_mul, MultiplicationOperator)
        self.set_rule(('.', ConstantOperator),
                      lambda s, o: s._rule_broadcast(
                          s, o, ConstantOperator, np.add),
                      AdditionOperator)
        self.set_rule(('.', ConstantOperator),
                      lambda s, o: s._rule_broadcast(
                          s, o, ConstantOperator, np.multiply),
                      MultiplicationOperator)

    def direct(self, input, output, operation=operation_assignment):
        if self.broadcast == 'rightward':
            operation(output.T, self.data.T)
        else:
            operation(output, self.data)

    @staticmethod
    def _rule_left(self, op):
        if op.commin is not None or op.commout is not None:
            return None
        return self.copy()

    @staticmethod
    def _rule_right(op, self):
        if op.commin is not None or op.commout is not None:
            return None
        if op.flags.shape_output == 'unconstrained':
            return None
        if self.flags.shape_output == 'explicit':
            data = self._as_strided(self.shapeout)
        elif op.flags.shape_input == 'explicit':
            data = self._as_strided(op.shapein)
        else:
            return None
        return ConstantOperator(op(data))

    @staticmethod
    def _rule_mul(self, op):
        if not isinstance(op, CompositionOperator) and not op.flags.linear:
            return
        s = DiagonalOperator(self.data, broadcast=self.broadcast)
        return CompositionOperator([s, op])

    @staticmethod
    def _rule_left_block(op, self):
        return

    @staticmethod
    def _rule_right_block(self, op, cls):
        if cls is CompositionOperator:
            return
        return BroadcastingBase._rule_right_block(self, op, cls)

    def __neg__(self):
        return ConstantOperator(
            -self.data, broadcast=self.broadcast, shapein=self.shapein,
            shapeout=self.shapeout, reshapein=self.reshapein,
            reshapeout=self.reshapeout, dtype=self.dtype)

    def __str__(self):
        return str(self.data)


@real
@linear
class ZeroOperator(ConstantOperator):
    """
    A subclass of ConstantOperator with data = 0.

    """
    def __init__(self, *args, **keywords):
        ConstantOperator.__init__(self, 0, **keywords)
        self.del_rule(('.', BlockOperator), MultiplicationOperator)
        self.del_rule(('.', ConstantOperator), MultiplicationOperator)
        self.del_rule(('.', Operator), MultiplicationOperator)
        self.set_rule('T', lambda s: ZeroOperator())
        self.set_rule(('.', Operator), lambda s, o: o.copy(), AdditionOperator)
        self.set_rule(('.', Operator), lambda s, o: s.copy(),
                      MultiplicationOperator)

    def direct(self, input, output, operation=operation_assignment):
        operation(output, 0)

    @staticmethod
    def _rule_left(self, op):
        if op.commin is not None or op.commout is not None:
            return None
        return ZeroOperator()

    @staticmethod
    def _rule_right(op, self):
        if op.commin is not None or op.commout is not None:
            return None
        if op.flags.linear:
            return ZeroOperator()
        return ConstantOperator._rule_right(op, self)

    def __neg__(self):
        return self


class ReductionOperator(Operator):
    """
    Reduction-along-axis operator.

    Parameters
    ----------
    func : ufunc or function
        Function used for the reduction. If the input is a ufunc, its 'reduce'
        method is used.
    axis : integer, optional
        Axis along which the reduction is performed. If None, all dimensions
        are collapsed.
    dtype : dtype, optional
        Reduction data type.
    skipna : boolean, optional
        If this is set to True, the reduction is done as if any NA elements
        were not counted in the array. The default, False, causes the NA values
        to propagate, so if any element in a set of elements being reduced is
        NA, the result will be NA.

    Example
    -------
    >>> op = ReductionOperator(np.nansum)
    >>> op([np.nan, 1, 2])
    array(3.0)

    """
    def __init__(self, func, axis=None, dtype=None, skipna=False, **keywords):
        if axis is None:
            keywords['shapeout'] = ()
        if isinstance(func, np.ufunc):
            if func.nin != 2:
                raise TypeError(
                    "The input ufunc '{0}' has {1} input argument. Expected nu"
                    "mber is 2.".format(func.__name__, func.nin))
            if func.nout != 1:
                raise TypeError(
                    "The input ufunc '{0}' has {1} output arguments. Expected "
                    "number is 1.".format(func.__name__, func.nout))
            if np.__version__ < '2':
                if axis is None:
                    direct = lambda x, out: func.reduce(x.flat, 0, dtype, out)
                else:
                    direct = lambda x, out: func.reduce(x, axis, dtype, out)
            else:
                direct = lambda x, out: func.reduce(x, axis, dtype, out,
                                                    skipna=skipna)
        elif isinstance(func, types.FunctionType):
            vars, junk, junk, junk = inspect.getargspec(func)
            if 'axis' not in vars:
                raise TypeError("The input function '{0}' does not have an 'ax"
                                "is' argument.".format(func.__name__))
            kw = {}
            if 'dtype' in vars:
                kw['dtype'] = dtype
            if 'skipna' in vars:
                kw['skipna'] = skipna
            if 'out' not in vars:
                def direct(x, out):
                    out[...] = func(x, axis=axis, **kw)
            else:
                direct = lambda x, out: func(x, axis=axis, out=out, **kw)
        self.axis = axis
        Operator.__init__(self, direct=direct, dtype=dtype, **keywords)

    def reshapein(self, shape):
        if self.axis == -1:
            return shape[:-1]
        return shape[:self.axis] + shape[self.axis+1:]

    def validatein(self, shape):
        if len(shape) == 0:
            raise TypeError('Cannot reduce on scalars.')
        if self.axis is None:
            return
        if len(shape) < (self.axis+1 if self.axis >= 0 else abs(self.axis)):
            raise ValueError('The input shape has an insufficient number of di'
                             'mensions.')


@linear
@square
class Variable(Operator):
    """
    Fake operator to represent a variable.

    """
    def __init__(self, name, shape=None):
        self.name = name
        Operator.__init__(self, shapein=shape)
        self.set_rule('T',
                      lambda s: VariableTranspose(self.name, self.shapein))
        self.set_rule(('.', Operator), self._rule_rcomp, CompositionOperator)

    @staticmethod
    def _rule_rcomp(self, other):
        raise TypeError('A variable cannot be composed with an operator.')

    def __mul__(self, other):
        if isinstance(other, Variable):
            return MultiplicationOperator([self, other])
        if isinstance(other, VariableTranspose):
            return CompositionOperator([self, other])
        if np.isscalar(other) or isinstance(other, HomothetyOperator) or \
           isinstance(other, (list, tuple, np.ndarray)) and \
           not isinstance(other, np.matrix):
            return CompositionOperator([other, self])
        try:
            other = asoperator(other)
        except TypeError:
            return NotImplemented
        return MultiplicationOperator([other, self])

    def __rmul__(self, other):
        try:
            other = asoperator(other)
        except TypeError:
            return NotImplemented
        if other.flags.linear:
            return CompositionOperator([other, self])
        return MultiplicationOperator([other, self])

    def __pow__(self, n):
        return po.nonlinear.PowerOperator(n)(self)

    def __str__(self):
        return self.name

    __repr__ = __str__


@linear
@square
class VariableTranspose(Operator):
    """
    Fake operator to represent a transposed variable.

    """
    def __init__(self, name, shape=None):
        self.name = name
        Operator.__init__(self, shapein=shape)
        self.set_rule('T', lambda s: Variable(self.name, self.shapein))
        self.set_rule((Operator, '.'), self._rule_lcomp, CompositionOperator)

    @staticmethod
    def _rule_lcomp(self, other):
        raise ValueError('An operator cannot be composed with a transposed var'
                         'iable.')

    def __mul__(self, other):
        if isinstance(other, VariableTranspose):
            raise TypeError('Transposed variables cannot be multiplied.')
        if isinstance(other, Variable):
            return CompositionOperator([self, other])
        if isscalarlike(other) or isinstance(other, HomothetyOperator):
            return CompositionOperator([other, self])
        if isinstance(other, np.ndarray) and not isinstance(other, np.matrix):
            return CompositionOperator([self, DiagonalOperator(other)])
        try:
            other = asoperator(other)
        except TypeError:
            return NotImplemented
        if not other.flags.linear:
            raise TypeError('Multiplying a transposed variable by a non-linear'
                            ' operator does not make sense.')
        return CompositionOperator([self, other])

    def __rmul__(self, other):
        if np.isscalar(other) or isinstance(other, HomothetyOperator):
            return CompositionOperator([self, other])
        raise TypeError('An operator cannot be composed with a transposed vari'
                        'able.')

    def __str__(self):
        return self.name + '.T'

    __repr__ = __str__


def _copy_direct(source, target):
    keywords = {}
    for attr in set(OPERATOR_ATTRIBUTES) - {
            'flags', 'reshapein', 'reshapeout', 'toshapein', 'toshapeout',
            'validatein', 'validateout'}:
        v = getattr(source, attr)
        keywords[attr] = v
    Operator.__init__(target, **keywords)
    return target


def _copy_direct_all(source, target):
    keywords = {}
    for attr in set(OPERATOR_ATTRIBUTES) - {'flags'}:
        v = getattr(source, attr)
        if attr in ('reshapein', 'reshapeout', 'toshapein', 'toshapeout',
                    'validatein', 'validateout'):
            if v == getattr(Operator, attr).__get__(source, type(source)):
                continue
        keywords[attr] = v
    Operator.__init__(target, **keywords)
    return target


def _copy_reverse(source, target):
    keywords = {}
    for attr in set(OPERATOR_ATTRIBUTES) - {
            'flags', 'reshapein', 'reshapeout', 'toshapein', 'toshapeout',
            'validatein', 'validateout'}:
        v = getattr(source, attr)
        keywords[_swap_inout(attr)] = v
    Operator.__init__(target, **keywords)
    return target


def _copy_reverse_all(source, target):
    keywords = {}
    for attr in set(OPERATOR_ATTRIBUTES) - {'flags'}:
        v = getattr(source, attr)
        if attr in ('reshapein', 'reshapeout', 'toshapein', 'toshapeout',
                    'validatein', 'validateout'):
            if v == getattr(Operator, attr).__get__(source, type(source)):
                continue
        keywords[_swap_inout(attr)] = v
    Operator.__init__(target, **keywords)
    return target


def _swap_inout(s):
    if s.endswith('in'):
        return s[:-2] + 'out'
    elif s.endswith('out'):
        return s[:-3] + 'in'
    return s


def asoperator(x, constant=False, **keywords):
    """
    Return input as an Operator.

    Parameters
    ----------
    x : object
        The input can be one of the following:
            - a callable (including ufuncs)
            - array_like (including matrices)
            - a numpy or python scalar
            - scipy.sparse.linalg.LinearOperator
    constant : boolean, optional
        If True, return a ConstantOperator instead of a HomothetyOperator for
        scalars. Default is False.
    flags : dictionary
        The operator flags.

    """
    if isinstance(x, Operator):
        return x

    if isinstance(x, np.ufunc):
        return Operator(x, **keywords)

    if np.isscalar(x) or isinstance(x, (list, tuple)):
        x = np.array(x)

    if isinstance(x, np.ndarray):
        if constant and not isinstance(x, np.matrix):
            return ConstantOperator(x, **keywords)
        if x.ndim == 0:
            return HomothetyOperator(x, **keywords)
        if x.ndim == 1:
            return DiagonalOperator(x, shapein=x.shape[-1], **keywords)
        return po.linear.DenseBlockDiagonalOperator(
            x, shapein=x.shape[:-2] + (x.shape[-1],), **keywords)

    if sp.issparse(x):
        return po.linear.SparseOperator(x, **keywords)

    if hasattr(x, 'matvec') and hasattr(x, 'rmatvec') and \
       hasattr(x, 'shape'):
        def direct(input, output):
            output[...] = x.matvec(input)

        def transpose(input, output):
            output[...] = x.rmatvec(input)
        keywords['flags'] = Operator.validate_flags(keywords.get('flags', {}),
                                                    linear=True)
        return Operator(direct=direct, transpose=transpose,
                        shapein=x.shape[1], shapeout=x.shape[0],
                        dtype=x.dtype, **keywords)

    if isinstance(x, collections.Callable):
        def direct(input, output):
            output[...] = x(input)
        keywords['flags'] = Operator.validate_flags(keywords.get('flags', {}),
                                                    inplace=True)
        return Operator(direct, **keywords)

    try:
        op = sp.linalg.aslinearoperator(x)
    except Exception as e:
        raise TypeError(e)
    return asoperator(op, **keywords)


def asoperator1d(x):
    x = asoperator(x)
    r = ReshapeOperator(x.shape[1], x.shapein)
    s = ReshapeOperator(x.shapeout, x.shape[0])
    return s * x * r

_pool = MemoryPool()
timer_operator = Timer(cumulative=True)
