#coding: utf-8
"""
The core module defines the Operator class. Operators are functions
which can be added, composed or multiplied by a scalar. See the
Operator docstring for more information.
"""

from __future__ import division

import copy
import inspect
import numpy as np
import operator
import scipy.sparse.linalg
import types

from collections import MutableMapping, MutableSequence, MutableSet, namedtuple
from itertools import izip
from .memory import empty, iscompatible, zeros, MemoryPool, MEMORY_ALIGNMENT
from .utils import (all_eq, first_is_not, inspect_special_values, isalias,
                    isclassattr, isscalar, merge_none, ndarraywrap,
                    operation_assignment, product, renumerate, strenum,
                    strshape, tointtuple, ufuncs)
from .utils.mpi import MPI
from .decorators import (linear, real, idempotent, involutary, square,
                         symmetric, inplace)

__all__ = [
    'Operator',
    'OperatorFlags',
    'AdditionOperator',
    'BlockColumnOperator',
    'BlockDiagonalOperator',
    'BlockRowOperator',
    'BlockSliceOperator',
    'BroadcastingOperator',
    'CompositionOperator',
    'ConstantOperator',
    'DenseOperator',
    'DiagonalOperator',
    'GroupOperator',
    'HomothetyOperator',
    'IdentityOperator',
    'MaskOperator',
    'MultiplicationOperator',
    'ReshapeOperator',
    'ReductionOperator',
    'ZeroOperator',
    'DirectOperatorFactory',
    'ReverseOperatorFactory',
    'asoperator',
    'I',
    'O',
]

OPERATOR_ATTRIBUTES = ['attrin', 'attrout', 'classin', 'classout', 'commin',
                       'commout', 'reshapein', 'reshapeout', 'shapein',
                       'shapeout', 'toshapein', 'toshapeout', 'validatein',
                       'validateout', 'dtype', 'flags']

class OperatorFlags(namedtuple('OperatorFlags',
                               [
                                'linear',
                                'square',     # shapein == shapeout
                                'real',       # o.C = o
                                'symmetric',  # o.T = o
                                'hermitian',  # o.H = o
                                'idempotent', # o * o = o
                                'involutary', # o * o = I
                                'orthogonal', # o * o.T = I
                                'unitary',    # o * o.H = I
                                'separable',  # o*[B1...Bn] = [o*B1...o*Bn]
                                'inplace',
                                'inplace_reduction',
                                'alignment_input', # op. requires aligned input
                                'alignment_output',# op. requires aligned output
                                'contiguous_input', # op. requires contig input
                                'contiguous_output',# op. requires contig output
                                'shape_input',
                                'shape_output',
                                ])):
    """ Informative flags about the operator. """
    def __new__(cls):
        t = 12*(False,) + (1, 1, False, False, '', '')
        return super(OperatorFlags, cls).__new__(cls, *t)

    def __str__(self):
        n = max(len(f) for f in self._fields)
        fields = [ '  ' + f.upper().ljust(n) + ' : ' for f in self._fields]
        return '\n'.join([f + str(v) for f,v in zip(fields,self)])

    def __repr__(self):
        n = max(len(f) for f in self._fields)
        fields = [ f.ljust(n) + '= ' for f in self._fields]
        return self.__class__.__name__ + '(\n  ' + ',\n  '.join(f + repr(v)
            for f,v in zip(fields,self)) + ')'


class OperatorRule(object):
    """
    Abstract class for operator rules.

    A operator rule is a relation that can be expressed by the sentence
    "'subjects' are 'predicate'". An instance of this class, when called with
    checks if the inputs are subjects to the rule, and returns the predicate
    if it is the case. Otherwise, it returns None.
    """
    def __init__(self, subjects, predicate):

        if not isinstance(subjects, str):
            raise TypeError("The input first argument '{0}' is not a string." \
                             .format(subjects))

        subjects_ = self._split_subject(subjects)
        if len(subjects_) == 0:
            raise ValueError('No rule subject is specified.')
        if not isinstance(self, OperatorUnaryRule) and len(subjects_) == 1:
            self.__class__ = OperatorUnaryRule
            self.__init__(subjects, predicate)
            return
        if not isinstance(self, OperatorBinaryRule) and len(subjects_) == 2:
            self.__class__ = OperatorBinaryRule
            self.__init__(subjects, predicate)
            return

        if '1' in subjects_:
            raise ValueError("'1' cannot be a subject.")

        if isinstance(predicate, str) and '{' in predicate:
            raise ValueError("Predicate cannot be a subclass.")

        self.subjects = subjects_
        self.predicate = predicate

    def __eq__(self, other):
        if not isinstance(other, OperatorRule):
            return NotImplemented
        if self.subjects != other.subjects:
            return False
        if isinstance(self.predicate, types.LambdaType):
            if type(self.predicate) is not type(other.predicate):
                return False
            return self.predicate.func_code is other.predicate.func_code
        if isinstance(self.predicate, str):
            return self.predicate == other.predicate
        return self.predicate  is other.predicate

    @staticmethod
    def _symbol2operator(op, symbol):
        if not isinstance(symbol, str):
            return symbol
        if  symbol == '1':
            return IdentityOperator()
        if symbol[0] == '{' and symbol[-1] == '}':
            return symbol[1:-1]
        if symbol == '.':
            return op
        try:
            return {'.C': op._C,
                    '.T': op._T,
                    '.H': op._H,
                    '.I': op._I}[symbol]
        except (KeyError):
            raise ValueError("Invalid symbol: '{0}'.".format(symbol))

    @classmethod
    def _split_subject(cls, subject):
        if isinstance(subject, (list, tuple)):
            return subject
        if not isinstance(subject, str):
            raise TypeError('The rule subject is not a string.')
        if len(subject) == 0:
            return []
        associated = '.IC', '.IT', '.IH', '.C', '.T', '.H', '.I', '.'
        for a in associated:
            if subject[:len(a)] == a:
                return [a] + cls._split_subject(subject[len(a):])
        if subject[0] == '{':
            try:
                pos = subject.index('}')
            except ValueError:
                raise ValueError("Invalid subject: no matching closing '}'.")
            return [subject[:pos+1]] + cls._split_subject(subject[pos+1:])

        raise ValueError("The subject {0} is not understood.".format(subject))

    def __str__(self):
        return '{0} = {1}'.format(''.join(self.subjects), self.predicate)

    __repr__ = __str__


class OperatorUnaryRule(OperatorRule):
    """
    Binary rule on operators.

    A operator unary rule is a relation that can be expressed by the sentence
    "'subject' is 'predicate'".

    Parameters
    ----------
    subject : str
        It defines the property of the operator for which the predicate holds:
            '.C' : the operator conjugate
            '.T' : the operator transpose
            '.H' : the operator adjoint
            '.I' : the operator adjoint
            '.IC' : the operator inverse-conjugate
            '.IT' : the operator inverse-transpose
            '.IH' : the operator inverse-adjoint

    predicate : function or str
        What is returned by the rule when is applies. It can be:
            '1' : the identity operator
            '.' : the operator itself
            or a callable of one argument.
    
    Example
    -------
    >>> rule = OperatorUnaryRule('.T', '.')
    >>> o = Operator()
    >>> oT = rule(o)
    >>> oT is o
    True
    
    """
    def __init__(self, subjects, predicate):
        super(OperatorUnaryRule, self).__init__(subjects, predicate)
        if len(self.subjects) != 1:
            raise ValueError('This is not a unary rule.')
        if self.subjects[0] == '.':
            raise ValueError('The subject cannot be the operator itself.')
        if callable(predicate) or predicate in ('.', '1'):
            return
        raise ValueError("Invalid predicate: '{0}'.".format(predicate))

    def __call__(self, reference):
        predicate = self._symbol2operator(reference, self.predicate)
        if predicate is None:
            return None
        if not isinstance(predicate, Operator ) and callable(predicate):
            predicate = predicate(reference)
        if not isinstance(predicate, Operator):
            raise TypeError('The predicate is not an operator.')
        return predicate


class OperatorBinaryRule(OperatorRule):
    """
    Binary rule on operators.

    A operator rule is a relation that can be expressed by the sentence
    "'subjects' are 'predicate'". An instance of this class, when called with
    two input arguments checks if the inputs are subjects to the rule, and
    returns the predicate if it is the case. Otherwise, it returns None.

    Parameters
    ----------
    subjects : str
        It defines the relationship between the two subjects that must be
        verified for the rule to apply. It is the concatenation of two
        expressions. One has to be '.' and stands for the reference subject.
        It determines if the reference operator is on the right or left hand
        side of the operator pair. The other expression constrains the other
        subject, which must be:
            '.' : the reference operator itself.
            '.C' : the conjugate of the reference object
            '.T' : the transpose of the reference object
            '.H' : the adjoint of the reference object
            '{...}' : an instance of the class '...'
            '{self}': an instance of the reference operator's class
        For instance, given a string '.C.', the rule will apply to the inputs
        o1 and o2 if o1 is o2.C. For a condition '.{DiagonalOperator}', the
        rule will apply if o2 is a DiagonalOperator instance.

    predicate : function or str
        If the two objects o1, o2, are subjects of the rule, the predicate
        will be returned. The predicate can also be '1', '.', '.C', '.T', '.H'
        of a callable of two arguments.
    
    Example
    -------
    >>> rule = OperatorBinaryRule('..', '.')
    >>> o = Operator()
    >>> rule(o, o) is o
    True
    >>> rule(o, IdentityOperator()) is None
    True
    
    """
    def __init__(self, subjects, predicate):
        super(OperatorBinaryRule, self).__init__(subjects, predicate)
        if len(self.subjects) != 2:
            raise ValueError('This is not a binary rule.')
        self.reference = 1 if self.subjects[1] == '.' else 0
        self.other = self.subjects[1-self.reference]

    def __call__(self, o1, o2):

        reference, other = (o1,o2) if self.reference == 0 else (o2,o1)
        subother = self._symbol2operator(reference, self.other)
        predicate = self._symbol2operator(reference, self.predicate)
        if predicate is None:
            return None

        if isinstance(subother, str):
            if subother == 'self':
                if not isinstance(other, reference.__class__):
                    return None
            elif subother == 'HomothetyOperator':
                if not isinstance(other, ZeroOperator) and subother not in (
                   c.__name__ for c in other.__class__.__mro__):
                    return None
            elif subother not in (c.__name__ for c in other.__class__.__mro__):
                return None
        elif other is not subother:
            return None

        if not isinstance(predicate, Operator) and callable(predicate):
            predicate = predicate(o1, o2)
        if predicate is None:
            return None
        if isinstance(predicate, (list,tuple)) and len(predicate) == 1:
            predicate = predicate[0]
        if not isinstance(predicate, Operator) and not (isinstance(predicate,
           (list,tuple)) and all(isinstance(o, Operator) for o in predicate)):
            raise TypeError("The predicate '{0}' is not an operator.".format(
                            predicate))
        return predicate


class Operator(object):
    """
    Operator top-level class.

    The operator class is a function factory.

    Attributes
    ----------
    attrin/attrout : dict or function
        If attrout is a dict, its items are added to the output. If it is
        a function, it takes the input attributes and returns the output
        attributes. The attrin attribute is only used in the reversed direction.
    classin/classout : ndarray subclass
        The classout attribute sets the output class. The classin attribute is
        only used in the reversed direction.
    commin/commout : mpi4py.Comm
        The commin and commout attributes store the MPI communicator for the in-
        put and output.
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
    flags : OperatorFlags
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
                 conjugate_=None, inverse=None, inverse_transpose=None,
                 inverse_adjoint=None, inverse_conjugate=None,
                 attrin={}, attrout={}, classin=None, classout=None,
                 commin=None, commout=None, reshapein=None, reshapeout=None,
                 shapein=None, shapeout=None, toshapein=None, toshapeout=None,
                 validatein=None, validateout=None, dtype=None, flags={},
                 name=None):
            
        for method, name_ in zip( \
            (direct, transpose, adjoint, conjugate_, inverse, inverse_transpose,
             inverse_adjoint, inverse_conjugate),
            ('direct', 'transpose', 'adjoint', 'conjugate_', 'inverse',
             'inverse_transpose', 'inverse_adjoint', 'inverse_conjugate')):
            if method is not None:
                if not hasattr(method, '__call__'):
                    raise TypeError("The method '%s' is not callable." % name_)
                # should also check that the method has at least two arguments
                setattr(self, name_, method)

        if self.transpose is None and self.adjoint is not None:
            def transpose(input, output):
                self.adjoint(input.conjugate(), output)
                output[...] = output.conjugate()
            self.transpose = transpose

        if self.adjoint is None and self.transpose is not None:
            def adjoint(input, output):
                self.transpose(input.conjugate(), output)
                output[...] = output.conjugate()

        if self.inverse is None:
            self.inverse_conjugate = None

        self._init_dtype(dtype)
        self._init_flags(flags)
        self._init_rules()
        self._init_name(name)
        self._init_inout(attrin, attrout, classin, classout, commin, commout,
                         reshapein, reshapeout, shapein, shapeout, toshapein,
                         toshapeout, validatein, validateout)

    dtype = None
    flags = OperatorFlags()
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
            raise ValueError("The operator '" + self.__name__ + "' does not hav"
                             "e an explicit shape.")
        return v.reshape(self.shapein)

    def toshapeout(self, v):
        """
        Reshape a vector into a multi-dimensional array compatible with
        the operator's output shape.

        """
        if self.shapeout is None:
            raise ValueError("The operator '" + self.__name__ + "' does not hav"
                             "e an explicit shape.")
        return v.reshape(self.shapeout)

    def propagate_attributes(self, cls, attr):
        """
        Propagate attributes according to operator's attrout. If the class
        changes, class attributes are removed if they are not class attributes
        of the new class.
        """
        if None not in (self.classout, cls) and self.classout is not cls:
            for a in attr.keys():
                if isclassattr(cls, a) and not isclassattr(self.classout, a):
                    del attr[a]
        if 'shape_global' in attr:
            del attr['shape_global']
        if isinstance(self.attrout, dict):
            for k, v in self.attrout.items():
                if isinstance(v, (MutableMapping, MutableSequence, MutableSet)):
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
            raise ValueError("The input shape '{0}' is incompatible with that o"
                "f {1}: '{2}'.".format(shapein, self.__name__, self.shapein))

    def validateout(self, shapeout):
        """
        Validate an output shape by raising a ValueError exception if it is
        invalid.

        """
        if self.shapeout is not None and self.shapeout != shapeout:
            raise ValueError("The output shape '{0}' is incompatible with that "
                "of {1}: '{2}'.".format(shapeout, self.__name__, self.shapeout))
    
    # for the next methods, the following always stand:
    #    - input and output are not in the memory pool
    #    - input and output are compatible with the operator's requirements
    #      in terms of shape, contiguity and alignment.
    direct = None
    def conjugate_(self, input, output):
        self.direct(input.conjugate(), output)
        output[...] = output.conjugate()
    transpose = None
    adjoint = None
    inverse = None
    def inverse_conjugate(self, input, output):
        self.inverse(input.conjugate(), output)
        output[...] = output.conjugate()
    inverse_transpose = None
    inverse_adjoint = None

    def __call__(self, x, out=None, preserve_input=True, propagate=True):

        if isinstance(x, Operator):
            return CompositionOperator([self, x])

        if self.direct is None:
            raise NotImplementedError('Call to ' + self.__name__ + ' is not imp'
                                      'lemented.')

        # get valid input and output
        i, i_, o, o_ = self._validate_arguments(x, out)

        # perform computation
        reuse_x = isinstance(x, np.ndarray) and not isalias(x, i) and \
                  not preserve_input
        reuse_out = isinstance(out, np.ndarray) and not isalias(out, i) \
                    and not isalias(out, o)

        with _pool.set_if(reuse_x, x):
            with _pool.set_if(reuse_out, out):
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
        # the iteration is sorted by key, so that attributes beginning with an
        # underscore are set first.
        for k in sorted(attr.keys()):
            setattr(out, k, attr[k])
        return out

    @property
    def shape(self):
        return (product(self.shapeout), product(self.shapein))

    @staticmethod
    def isalias(array1, array2):
        return array1.__array_interface__['data'][0] == \
               array2.__array_interface__['data'][0]

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
            raise ValueError("The operator's input shape is not explicit. Speci"
                             "fy it with the 'shapein' keyword.")
        if shapeout is None:
            raise ValueError("The operator's output shape is not explicit. Spec"
                             "ify it with the 'shapeout' keyword.")
        m, n = product(shapeout), product(shapein)
        d = np.empty((n,m), self.dtype)

        if not inplace or not self.flags.inplace:
            v = zeros(n, self.dtype)
            if self.flags.alignment_output == 1:
                for i in xrange(n):
                    v[i] = 1
                    o = d[i,:].reshape(shapeout)
                    self.direct(v.reshape(shapein), o)
                    v[i] = 0
            else:
                o = empty(shapeout, self.dtype)
                for i in xrange(n):
                    v[i] = 1
                    self.direct(v.reshape(shapein), o)
                    d[i,:] = o.ravel()
                    v[i] = 0
            return d.T

        # test in-place mechanism
        u = empty(max(m,n), self.dtype)
        v = u[:n]
        w = u[:m]
        for i in xrange(n):
            v[:] = 0
            v[i] = 1
            self.direct(v.reshape(shapein), w.reshape(shapeout))
            d[i,:] = w
        return d.T

    def matvec(self, x, out=None):

        assert not isinstance(x, np.ndarray) or x.flags.contiguous
        assert out is None or \
               isinstance(out, np.ndarray) and out.flags.contiguous
        x = self.toshapein(x)
        if out is not None:
            out = self.toshapeout(out)
        out = self.__call__(x, out=out, propagate=False)

        return out.ravel()

    def rmatvec(self, x, out=None):
        return self.T.matvec(x, out=out)

    def set_rule(self, subjects, predicate, operation=None, globals=None):
        """
        Add a rule to the rule list, taking care of duplicates and priorities.
        Class-matching rules have a lower priority than the others.

        Parameters
        ----------
        subjects : str
            See OperatorUnaryRule and OperatorBinaryRule documentation.
        predicate : str
            See OperatorUnaryRule and OperatorBinaryRule documentation.
        operation : CompositeOperator sub class
            Operation to which applies the rule. It can be: CompositionOperator,
            AdditionOperator and MultiplicationOperator. For unary rules,
            the value must be None.
        globals : dict, optional
            Dictionary containing the operator classes used in class-matching
            rules. It is required for classes not from pyoperators.core and for
            which more than one class-matching rule is set.
        """
        rule = OperatorRule(subjects, predicate)

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
                self.rules[operation] = {'left':[], 'right':[]}
            rules = self.rules[operation]['left' if rule.reference == 0 else \
                                          'right']
        ids = [r.subjects for r in rules]

        # first, try to override existing rule
        try:
            index = ids.index(rule.subjects)
            rules[index] = rule
            return
        except ValueError:
            pass

        if len(rule.subjects) == 1 or not rule.other.startswith('{'):
            rules.insert(0, rule)
            return

        # search for subclass rules
        try:
            index = [r.other[0] for r in rules].index('{')
        except ValueError:
            rules.append(rule)
            return

        # insert the rule after more specific ones
        cls = type(self) if rule.other[1:-1] == 'self' else \
              eval(rule.other[1:-1], globals)
        classes = [ r.other[1:-1] for r in rules[index:] ]
        classes = [ cls if r == 'self' else eval(r, globals) for r in classes ]
        is_subclass = [ issubclass(cls, c) for c in classes ]
        is_supclass = [ issubclass(c, cls) for c in classes ]
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
        subjects = OperatorRule._split_subject(subjects)
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

    def conjugate(self):
        """ Return the complex-conjugate of the operator. Same as '.C'. """
        return self.C

    def copy(self):
        """ Return a copy of the operator. """
        return copy.copy(self)

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
        if None in self.rules:
            rules = dict((r.subjects[0],r) for r in self.rules[None])
        else:
            rules = {}

        flags = self.flags

        if flags.real:
            C = self
        elif '.C' in rules:
            C = rules['.C'](self)
        else:
            C = DirectOperatorFactory(Operator, self, direct=self.conjugate_,
                                      name=self.__name__ + '.C',
                                      flags={'linear':flags.linear,
                                             'symmetric':flags.symmetric,
                                             'hermitian':flags.hermitian,
                                             'idempotent':flags.idempotent,
                                             'involutary':flags.involutary,
                                             'orthogonal':flags.orthogonal,
                                             'unitary':flags.unitary})

        if flags.symmetric:
            T = self
        elif '.T' in rules:
            T = rules['.T'](self)
        else:
            T = ReverseOperatorFactory(Operator, self, direct=self.transpose,
                                       name=self.__name__ + '.T',
                                       flags={'linear':flags.linear,
                                              'idempotent':flags.idempotent,
                                              'involutary':flags.involutary,
                                              'orthogonal':flags.orthogonal,
                                              'unitary':flags.unitary})

        if flags.hermitian:
            H = self
        elif '.H' in rules:
            H = rules['.H'](self)
        elif flags.real:
            H = T
        elif flags.symmetric:
            H = C
        else:
            H = ReverseOperatorFactory(Operator, self, direct=self.adjoint,
                                       name=self.__name__ + '.H',
                                       flags={'linear':flags.linear,
                                              'idempotent':flags.idempotent,
                                              'involutary':flags.involutary,
                                              'orthogonal':flags.orthogonal,
                                              'unitary':flags.unitary})

        if flags.involutary:
            I = self
        elif '.I' in rules:
            I = rules['.I'](self)
        elif flags.orthogonal:
            I = T
        elif flags.unitary:
            I = H
        else:
            I = ReverseOperatorFactory(Operator, self, direct=self.inverse,
                                       name=self.__name__ + '.I',
                                       flags={'idempotent':flags.idempotent,
                                              'involutary':flags.involutary,
                                              'orthogonal':flags.orthogonal,
                                              'unitary':flags.unitary})

        if flags.real:
            IC = I
        elif '.IC' in rules:
            IC = rules['.IC'](self)
        elif flags.orthogonal:
            IC = H
        elif flags.unitary:
            IC = T
        elif flags.involutary:
            IC = C
        else:
            IC = ReverseOperatorFactory(Operator, self,
                                        direct=self.inverse_conjugate,
                                        name=self.__name__ + '.I.C',
                                        flags={'idempotent':flags.idempotent,
                                               'involutary':flags.involutary,
                                               'orthogonal':flags.orthogonal,
                                               'unitary':flags.unitary})

        if flags.orthogonal:
            IT = self
        elif flags.symmetric:
            IT = I
        elif flags.unitary:
            IT = C
        elif flags.involutary:
            IT = T
        elif '.IT' in rules:
            IT = rules['.IT'](self)
        else:
            IT = DirectOperatorFactory(Operator, self,
                                       direct=self.inverse_transpose,
                                       name=self.__name__ + '.I.T',
                                       flags={'idempotent':flags.idempotent,
                                              'involutary':flags.involutary,
                                              'orthogonal':flags.orthogonal,
                                              'unitary':flags.unitary})

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
        elif '.IH' in rules:
            IH = rules['.IH'](self)
        else:
            IH = DirectOperatorFactory(Operator, self,
                                       direct=self.inverse_adjoint,
                                       name=self.__name__ + '.I.H',
                                       flags={'idempotent':flags.idempotent,
                                              'involutary':flags.involutary,
                                              'orthogonal':flags.orthogonal,
                                              'unitary':flags.unitary})

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

        if isinstance(self.direct, np.ufunc):
            if self.direct.nin != 1 or self.direct.nout != 1:
                raise TypeError('An ufunc with several inputs or outputs cannot'
                                ' be converted to an Operator.')
            real = True
            for t in self.direct.types:
                i, o = t[0], t[3]
                if o not in 'FDG':
                    continue
                if i not in 'FDG':
                    real = False
                    break
            if real:
                self._set_flags('real')
            self._set_flags('inplace')
            self._set_flags('square')
            self._set_flags('separable')
            if self.flags.inplace_reduction:
                raise ValueError('Ufuncs do not handle inplace reductions.')
        else:
            if isinstance(self.direct, types.MethodType):
                d = self.direct.im_func
            else:
                d = self.direct
            if isinstance(flags, (dict, str)) and 'inplace_reduction' not in \
               flags or isinstance(flags, OperatorFlags):
                if d is not None and 'operation' in d.func_code.co_varnames:
                    self._set_flags('inplace_reduction')
            if self.flags.inplace_reduction:
                if d is not None and 'operation' not in d.func_code.co_varnames:
                    raise TypeError("The direct method of an inplace-reduction "
                            "operator must have an 'operation' keyword.")

        if self.flags.inplace:
            alignment = max(self.flags.alignment_input,
                            self.flags.alignment_output)
            contiguous = max(self.flags.contiguous_input,
                             self.flags.contiguous_output)
            self._set_flags({'alignment_input':alignment,
                             'alignment_output': alignment,
                             'contiguous_input': contiguous,
                             'contiguous_output': contiguous})

    def _init_rules(self):
        """ Translate flags into rules. """
        if self.rules is None:
            self.rules = {}

        if self.flags.real:
            self.set_rule('.C', '.')
        if self.flags.symmetric:
            self.set_rule('.T', '.')
        if self.flags.hermitian:
            self.set_rule('.H', '.')
        if self.flags.involutary:
            self.set_rule('.I', '.')

        self.set_rule('.I.', '1', CompositionOperator)
        if self.flags.orthogonal:
            self.set_rule('.T.', '1', CompositionOperator)
        if self.flags.unitary:
            self.set_rule('.H.', '1', CompositionOperator)
        if self.flags.idempotent:
            self.set_rule('..', '.', CompositionOperator)

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
            raise TypeError("The 'attrin' keyword should be a dictionary or a f"
                            "unction.")
        if isinstance(attrout, (dict, types.FunctionType, types.MethodType)):
            if not isinstance(attrout, dict) or len(attrout) > 0:
                self.attrout = attrout
        else:
            raise TypeError("The 'attrout' keyword should be a dictionary or a "
                            "function.")
        if type(classin) is type and issubclass(classin, np.ndarray):
            self.classin = classin
        elif classin is not None:
            raise TypeError("The 'classin' keyword is not an ndarray subclass.")
        if type(classout) is type and issubclass(classout, np.ndarray):
            self.classout = classout
        elif classout is not None:
            raise TypeError("The 'classout' keyword is not an ndarray subclass.")
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

        shapein = tointtuple(shapein)
        shapeout = tointtuple(shapeout)
        self.shapein = shapein
        self.shapeout = shapeout
        if shapein is not None:
            shapeout = tointtuple(self.reshapein(shapein))
            if self.shapeout is None:
                self.shapeout = shapeout
        if shapeout is not None:
            shapein = tointtuple(self.reshapeout(shapeout))
            if self.shapein is None:
                self.shapein = shapein

        if shapein is not None:
            self.validatein(shapein)
        if shapeout is not None:
            self.validateout(shapeout)

        if self.shapein is not None and self.shapein == self.shapeout:
            self._set_flags('square')

        if self.flags.square:
            if self.shapein is None:
                self.shapein = self.shapeout
            else:
                self.shapeout = self.shapein
            self.reshapein = lambda x:x
            self.reshapeout = self.reshapein
            self.validatein = self.validatein or self.validateout
            self.validateout = self.validatein
            if self.toshapein.im_func is Operator.toshapein.im_func and \
               self.toshapeout.im_func is not Operator.toshapeout.im_func:
                self.toshapein = self.toshapeout
            else:
                self.toshapeout = self.toshapein

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

        flag_is = 'explicit' if self.shapein is not None else 'implicit' \
            if self.reshapeout != Operator.reshapeout.__get__(self, type(self))\
            else 'unconstrained'
        flag_os = 'explicit' if self.shapeout is not None else 'implicit' \
            if self.reshapein != Operator.reshapein.__get__(self, type(self)) \
            else 'unconstrained'
        self._set_flags(shape_input=flag_is, shape_output=flag_os)

        if flag_is == 'explicit':
            self.reshapeout = Operator.reshapeout.__get__(self, type(self))
            self.validatein = Operator.validatein.__get__(self, type(self))
        if flag_os == 'explicit':
            if self.flags.square:
                self.reshapein = self.reshapeout
                self.validateout = self.validatein
            else:
                self.reshapein = Operator.reshapein.__get__(self, type(self))
                self.validateout = Operator.validateout.__get__(self,type(self))
                    
    def _init_name(self, name):
        """ Set operator's __name__ attribute. """
        if name is None:
            if self.__class__ != 'Operator':
                name = self.__class__.__name__
            elif self.direct is not None and self.direct.__name__ not in \
                 ('<lambda>', 'direct'):
                name = self.direct.__name__
            else:
                name = 'Operator'
        self.__name__ = name

    def _set_flags(self, flags=None, **keywords):
        """ Set flags to an Operator. """
        if flags is None:
            flags = keywords
        if isinstance(flags, OperatorFlags):
            self.flags = flags
            return
        flags = self.validate_flags(flags)
        f = [k for k,v in flags.items() if v]
        if 'symmetric' in f or 'hermitian' in f or 'orthogonal' in f or \
           'unitary' in f:
            flags['linear'] = flags['square'] = True
        if 'orthogonal' in f:
            flags['real'] = True
        if 'involutary' in f:
            flags['square'] = True
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
           self.flags.alignment_input, self.flags.contiguous_input):
            if output is not None and self.flags.inplace and iscompatible(
               output, input.shape, dtype, self.flags.alignment_input,
               self.flags.contiguous_input):
                buf = output
            else:
                input_ = _pool.extract(input.shape, dtype,
                    self.flags.alignment_input, self.flags.contiguous_input)
                buf = input_
            input, input[...] = _pool.view(buf, input.shape, dtype), input

        # check compatibility of provided output
        if output is not None:
            if not isinstance(output, np.ndarray):
                raise TypeError('The output argument is not an ndarray.')
            output = output.view(np.ndarray)
            if output.dtype != dtype:
                raise ValueError("The output has an invalid dtype '{0}'. Expect"
                    "ed dtype is '{1}'.".format(output.dtype, dtype))

            # if the output does not fulfill the operator's alignment &
            # contiguity requirements, or if the operator is out-of-place and
            # an in-place operation is required, let's use a temporary buffer
            if not iscompatible(output, output.shape, dtype,
               self.flags.alignment_output, self.flags.contiguous_output) or \
               isalias(input, output) and not self.flags.inplace:
                output_ = _pool.extract(output.shape, dtype,
                    self.flags.alignment_output, self.flags.contiguous_output)
                output = _pool.view(output_, output.shape, dtype)
            shapeout = output.shape
        else:
            shapeout = None

        shapein, shapeout = self._validate_shapes(input.shape, shapeout)

        # if the output is not provided, allocate it
        if output is None:
            if self.flags.shape_input == 'implicit' and \
               self.flags.shape_output == 'unconstrained':
                raise ValueError('The output shape of an implicit input shape a'
                    'nd unconstrained output shape operator cannot be inferred.'
                    )
            if shapeout is None:
                shapeout = input.shape
            output = empty(shapeout, dtype, description="for {0}'s output.".format(
                           self.__name__))
        return input, input_, output, output_

    @staticmethod
    def validate_flags(flags, **keywords):
        """ Return flags as a dictionary. """
        if flags is None:
            return keywords
        if isinstance(flags, dict):
            flags = flags.copy()
        elif isinstance(flags, OperatorFlags):
            flags = dict((k,v) for k,v in zip(OperatorFlags._fields, flags))
        elif isinstance(flags, (list, tuple, str)):
            if isinstance(flags, str):
                flags = [f.strip() for f in flags.split(',')]
            flags = dict((f,True) for f in flags)
        else:
            raise TypeError("The operator flags have an invalid type '{0}'.".
                            format(flags))
        flags.update(keywords)
        if any(not isinstance(f, str) for f in flags):
            raise TypeError("Invalid type for the operator flags: {0}." \
                            .format(flags))
        if any(f not in OperatorFlags._fields for f in flags):
            raise ValueError("Invalid operator flags '{0}'. The properties "
                "must be one of the following: ".format(flags.keys()) + \
                strenum(OperatorFlags._fields) + '.')
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
            raise ValueError("The specified input shape '{0}' is incompatible w"
                "ith the expected one '{1}'.".format(shapein, shapein_))
        if None not in (shapeout, shapeout_) and shapeout != shapeout_:
            raise ValueError("The specified output shape '{0}' is incompatible "
                "with the expected one '{1}'.".format(shapeout, shapeout_))

        return (first_is_not([shapein, shapein_], None),
                first_is_not([shapeout, shapeout_], None))

    def __truediv__(self, other):
        return CompositionOperator([self, asoperator(other).I])
    __div__ = __truediv__

    def __rtruediv__(self, other):
        return CompositionOperator([other, self.I])
    __rdiv__ = __rtruediv__

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return self.matvec(other)
        # ensure that A * A is A if A is idempotent
        if self.flags.idempotent and self is other:
            return self
        return CompositionOperator([self, other])

    def __rmul__(self, other):
        if not isscalar(other):
            raise NotImplementedError("It is not possible to multiply '" + \
                str(type(other)) + "' with an Operator.")
        return CompositionOperator([other, self])

    def __imul__(self, other):
        # ensure that A * A is A if A is idempotent
        if self.flags.idempotent and self is other:
            return self
        return CompositionOperator([self, other])

    def __add__(self, other):
        return AdditionOperator([self, other])

    def __radd__(self, other):
        return AdditionOperator([other, self])

    def __iadd__(self, other):
        return AdditionOperator([self, other])

    def __sub__(self, other):
        return AdditionOperator([self, -other])

    def __rsub__(self, other):
        return AdditionOperator([other, -self])

    def __isub__(self, other):
        return AdditionOperator([self, -other])

    def __neg__(self):
        return HomothetyOperator(-1) * self

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        d1 = self.__dict__.copy()
        d2 = other.__dict__.copy()
        for k in '_C', '_T', '_H', '_I', '_D':
            if k in d1: del d1[k]
            if k in d2: del d2[k]
        return all_eq(d1, d2)

    def __str__(self):
        if self.shapein is not None or self.shapeout is not None:
            shapein = '?' if self.shapein is None else strshape(self.shapein)
            shapeout = '?' if self.shapeout is None else strshape(self.shapeout)
            if self.flags.square and self.shapein is not None and \
               len(self.shapein) > 1:
                s = shapein + '²'
            else:
                s = shapeout + 'x' + shapein
            s += ' '
        else:
            s = ''
        if hasattr(self, '__name__'):
            s += self.__name__
        else:
            s += type(self).__name__ + '[not initialized]'
        return s

    def __repr__(self):
        a = []
        init = getattr(self, '__init_original__', self.__init__)
        vars, args, keywords, defaults = inspect.getargspec(init)
            
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
            nargs = len(vars) - (len(defaults) if defaults is not None else 0)
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
                s += val.ndim * ']' +  ', dtype={0})'.format(val.dtype)
            elif var == 'dtype':
                s = str(val)
            else:
                s = repr(val)

            if ivar < nargs:
                a += [ s ]
            else:
                a += [var + '=' + s]
        name = self.__name__ if hasattr(self, '__name__') else \
               type(self).__name__ + '[not initialized]'
        return name + '(' + ', '.join(a) + ')'


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
    Composites can morph into their single operand during the call to
    CompositeOperator.__init__, As a consequence, one should make sure to return
    right after the call in the parent __init__ method.

    class MyCompositeOperator(CompositeOperator):
        def __init__(self, operands, **keywords):
            ...
            CompositeOperator.__init__(self, operands)
            if not isinstance()

    """
    def __init__(self, operands, dtype=None, **keywords):
        operands = self._validate_operands(operands)
        self._validate_comm(operands)
        if dtype is None:
            dtype = self._find_common_type(o.dtype for o in operands)
        self.operands = operands
        Operator.__init__(self, dtype=dtype, **keywords)
        self.propagate_commin(self.commin)
        self.propagate_commout(self.commout)

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
        if not isinstance(operands, (list, tuple)):
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
        elif isinstance(self, (BlockDiagonalOperator, BlockSliceOperator)):
            op = ' ⊕ '
        else:
            op = ' * '

        # parentheses for AdditionOperator and BlockDiagonalOperator
        operands = ['({0})'.format(o) if isinstance(o, (AdditionOperator,
                    BlockDiagonalOperator)) else str(o) for o in self.operands]

        # some special cases
        if isinstance(self, BlockDiagonalOperator) and  len(operands) > 2:
            operands = [operands[0], '...', operands[-1]]
        elif isinstance(self, CompositionOperator) and \
           isinstance(self.operands[0], HomothetyOperator):
            # remove trailing 'I'
            operands[0] = operands[0][:-1]
            if self.operands[0].data == -1:
                operands[0] += '1'

        return op.join(operands)

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
        CompositeOperator.__init__(self, operands, **keywords)
        if not isinstance(self, CommutativeCompositeOperator):
            return
        self.set_rule('.{Operator}', lambda s,o: type(s)(s.operands + [o]),
                      type(self))
        self.set_rule('.{self}', lambda s,o: type(s)(s.operands + o.operands),
                      type(self))
        self.operation = operation

    def direct(self, input, output):
        operands = list(self.operands)
        assert len(operands) > 1

        # we need a temporary buffer if all operands can do inplace reductions
        # except no more than one, which is move as first operand
        try:
            ir = [o.flags.inplace_reduction for o in operands]
            index = ir.index(False)
            operands[0], operands[index] = operands[index], operands[0]
            need_temporary = ir.count(False) > 1
        except ValueError:
            need_temporary = False

        operands[0].direct(input, output)
        ii = 0
        with _pool.get_if(need_temporary, output.shape, output.dtype) as buf:
            for op in operands[1:]:
                if op.flags.inplace_reduction:
                    op.direct(input, output, operation=self.operation)
                else:
                    op.direct(input, buf)
                    self.operation(output, buf)
                ii += 1

    def propagate_attributes(self, cls, attr):
        return Operator.propagate_attributes(self, cls, attr)

    def _apply_rules(self, ops):
        if len(ops) <= 1:
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
                        new_ops = rule(ops[i], ops[j])
                        if new_ops is None:
                            continue
                        del ops[j]
                        if j < i:
                            i -= 1
                        ops[i] = new_ops
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
        i = [i for i,o in enumerate(ops) if isinstance(o, HomothetyOperator)]
        if len(i) > 0:
            ops.insert(0, ops[i[0]])
            del ops[i[0]+1]
            if ops[0].data == 0 and len(ops) > 1:
                del ops[0]
        return ops

    @classmethod
    def _get_attributes(cls, operands, **keywords):
        attr = {
            'attrin':first_is_not((o.attrin for o in operands), None),
            'attrout':first_is_not((o.attrout for o in operands), None),
            'classin':first_is_not((o.classin for o in operands), None),
            'classout':first_is_not((o.classout for o in operands), None),
            'commin':first_is_not((o.commin for o in operands), None),
            'commout':first_is_not((o.commout for o in operands), None),
            'dtype':cls._find_common_type(o.dtype for o in operands),
            'flags':cls._merge_flags(operands),
            'reshapein':cls._merge_reshapein(operands),
            'reshapeout':cls._merge_reshapeout(operands),
            'shapein':cls._merge_shape((o.shapein for o in operands), 'in'),
            'shapeout':cls._merge_shape((o.shapeout for o in operands), 'out'),
            'toshapein':first_is_not((o.toshapein for o in operands), None),
            'toshapeout':first_is_not((o.toshapeout for o in operands), None),
            'validatein':first_is_not((o.validatein for o in operands), None),
            'validateout':first_is_not((o.validateout for o in operands), None),
        }
        for k, v in keywords.items():
            if k is not 'flags':
                attr[k] = v
        attr['flags'].update(Operator.validate_flags(keywords.get('flags', {})))
        return attr

    @staticmethod
    def _merge_flags(operands):
        return {
            'real':all(o.flags.real for o in operands),
            'alignment_input':max(o.flags.alignment_input for o in operands),
            'alignment_output':max(o.flags.alignment_output for o in operands),
            'contiguous_input':any(o.flags.contiguous_input for o in operands),
            'contiguous_output':any(o.flags.contiguous_output for o in operands)
        }

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
        operands = self._validate_operands(operands, constant=True)
        operands = self._apply_rules(operands)
        if len(operands) == 1:
            self.__class__ = operands[0].__class__
            self.__dict__ = operands[0].__dict__.copy()
            return
        CommutativeCompositeOperator.__init__(self, operands, operator.iadd,
                                              **keywords)
        self.set_rule('.T', lambda s: type(s)([m.T for m in s.operands]))
        self.set_rule('.H', lambda s: type(s)([m.H for m in s.operands]))
        self.set_rule('.C', lambda s: type(s)([m.C for m in s.operands]))

    @staticmethod
    def _merge_flags(operands):
        flags = CommutativeCompositeOperator._merge_flags(operands)
        flags.update({
            'linear':all(op.flags.linear for op in operands),
            'separable':all(o.flags.separable for o in operands),
            'square':any(o.flags.square for o in operands),
            'symmetric':all(op.flags.symmetric for op in operands),
            'hermitian':all(op.flags.symmetric for op in operands)})
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
        operands = self._apply_rules(operands)
        if len(operands) == 1:
            self.__class__ = operands[0].__class__
            self.__dict__ = operands[0].__dict__.copy()
            return
        CommutativeCompositeOperator.__init__(self, operands, operator.imul,
                                              **keywords)
        self.set_rule('.C', lambda s: type(s)([m.C for m in s.operands]))

    @staticmethod
    def _merge_flags(operands):
        flags = CommutativeCompositeOperator._merge_flags(operands)
        flags.update({
            'separable':all(o.flags.separable for o in operands),
            'square':any(o.flags.square for o in operands)})
        return flags


@square
class BlockSliceOperator(CommutativeCompositeOperator):
    """
    Class for multiple disjoint slices.

    The elements of the input not included in the slices are copied over
    to the output. This is due to fact that is not easy to derive the complement
    of a set of slices. To set those values to zeros, you might use MaskOperator
    or write a custom operator.
    Currently, there is no check to verify that the slices are disjoint.
    Non-disjoint slices can lead to unexpected results.

    Examples
    --------
    >>> op = BlockSliceOperator(HomothetyOperator(3), slice(None,None,2))
    >>> op(np.ones(6))
    array([ 3.,  1.,  3.,  1.,  3.,  1.])

    >>> op = BlockSliceOperator([ConstantOperator(1), ConstantOperator(2)],
                                ([slice(0,2), slice(0,2)], 
                                 [slice(2,4), slice(2,4)]))
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
        if not isinstance(slices, (list, tuple, slice)):
            raise TypeError('Invalid input slices.')
        if isinstance(slices, slice):
            slices = (slices,)
        if len(operands) != len(slices):
            raise ValueError("The number of slices '{0}' is not equal to the nu"
                "mber of operands '{1}'.".format(len(slices), len(operands)))

        keywords = self._get_attributes(operands, **keywords)
        CommutativeCompositeOperator.__init__(self, operands, **keywords)
        self.slices = tuple(slices)
        self.set_rule('.C', lambda s: 
                      BlockSliceOperator([op.C for op in s.operands], s.slices))
        self.set_rule('.T', lambda s: 
                      BlockSliceOperator([op.T for op in s.operands], s.slices))
        self.set_rule('.H', lambda s: 
                      BlockSliceOperator([op.H for op in s.operands], s.slices))

    def direct(self, input, output):
        if not isalias(input, output):
            output[...] = input
        for s, op in zip(self.slices, self.operands):
            i = input[s]
            o = output[s]
            with _pool.copy_if(i, op.flags.alignment_input,
                               op.flags.contiguous_input) as i:
                with _pool.copy_if(o, op.flags.alignment_output,
                                   op.flags.contiguous_output) as o:
                    op.direct(i, o)

    @classmethod
    def _get_attributes(cls, operands, **keywords):
        attr = {
            'dtype':cls._find_common_type(o.dtype for o in operands),
            'flags':cls._merge_flags(operands),
        }
        for k, v in keywords.items():
            if k is not 'flags':
                attr[k] = v
        attr['flags'].update(Operator.validate_flags(keywords.get('flags', {})))
        return attr

    @staticmethod
    def _merge_flags(operands):
        flags = CommutativeCompositeOperator._merge_flags(operands)
        flags.update({
            'linear':all(op.flags.linear for op in operands),
            'symmetric':all(op.flags.symmetric for op in operands),
            'hermitian':all(op.flags.hermitian for op in operands),
            'inplace':all(op.flags.inplace for op in operands)})
        return flags


class NonCommutativeCompositeOperator(CompositeOperator):
    """
    Abstract class for non-commutative composite operators, such as
    the composition.

    """
    def _apply_rules(self, ops):
        if len(ops) <= 1:
            return ops
        cls = type(self)
        i = len(ops) - 2

        # loop over the len(ops)-1 pairs of operands
        while i >= 0:
            
            o1 = ops[i]
            o2 = ops[i+1]
            rules1 = o1.rules[cls]['left'] if cls in o1.rules else []
            rules2 = o2.rules[cls]['right'] if cls in o2.rules else []

            # subclasses rules have a higher priority than those of superclasses
            if cls._ge_operator(o1, o2):
                rules = rules1 + rules2
            else:
                rules = rules2 + rules1

            consumed = False
            for rule in rules:
                new_ops = rule(o1, o2)
                if new_ops is None:
                    continue
                consumed = True
                if isinstance(new_ops, tuple):
                    if len(new_ops) != 2:
                        raise NotImplementedError()
                    ops[i], ops[i+1] = new_ops
                    i += 1
                    break
                cls._merge(new_ops, o1, o2)
                del ops[i+1]
                ops[i] = new_ops
                break

            if consumed and i < len(ops) - 1:
                continue

            i -= 1

        return ops

    @staticmethod
    def _ge_operator(o1, o2):
        """
        Return true if the first operator has a higher priority, i.e. if it
        subclasses the second argument class.
        """
        t1 = type(o1)
        t2 = type(o2)
        return issubclass(t1, t2) and t1 is not t2


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
        if len(operands) == 1:
            self.__class__ = operands[0].__class__
            self.__dict__ = operands[0].__dict__.copy()
            return
        keywords = self._get_attributes(operands, **keywords)
        NonCommutativeCompositeOperator.__init__(self, operands, **keywords)
        self._info = {}
        self.set_rule('.C', lambda s:type(s)([m.C for m in s.operands]))
        self.set_rule('.T', lambda s:type(s)([m.T for m in s.operands[::-1]]))
        self.set_rule('.H', lambda s:type(s)([m.H for m in s.operands[::-1]]))
        self.set_rule('.I', lambda s:type(s)([m.I for m in s.operands[::-1]]))
        self.set_rule('.IC',lambda s:type(s)([m.I.C for m in s.operands[::-1]]))
        self.set_rule('.IT', lambda s:type(s)([m.I.T for m in s.operands]))
        self.set_rule('.IH', lambda s:type(s)([m.I.H for m in s.operands]))
        self.set_rule('.{self}', lambda s,o: type(s)(s.operands + o.operands),
                      CompositionOperator)
        self.set_rule('.{Operator}', lambda s,o: type(s)(s.operands + [o]),
                      CompositionOperator)
        self.set_rule('{Operator}.', lambda o,s: type(s)([o] + s.operands),
                      CompositionOperator)

    def direct(self, input, output, operation=operation_assignment,
               preserve_input=True):

        preserve_input &= not isalias(input, output)
        preserve_output = operation is not operation_assignment

        shapeouts, dtypes, ninplaces, bufsizes, alignments, contiguouss  = \
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
        for igroup, (ninplace, bufsize, alignment, contiguous) in renumerate(
            zip(ninplaces, bufsizes, alignments, contiguouss)):

            if igroup != ngroups - 1:

                # get output for the current outplace operator if possible
                reuse_output = not preserve_output and (igroup % 2 == 0) and \
                    iscompatible(output, bufsize, np.int8, alignment,
                    contiguous) and not isalias(output, i) or igroup == 0
                if reuse_output:
                    o_ = output
                else:
                    o_ = _pool.extract(bufsize, np.int8, alignment, contiguous)
                    _pool.add(output)
                o = _pool.view(o_, shapeouts[iop], dtypes[iop])
                op = self.operands[iop]

                # perform out-of place operation
                if iop == 0 and self.flags.inplace_reduction:
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

    def _get_info(self, input, output, preserve_input):
        """
        Given the context in which the composition is taking place:
            1) input and output shape, dtype, alignment and contiguity
            2) in-place or out-of-place composition
            3) whether the input should be preserved,

        the routine returns the requirements for the intermediate buffers of the
        composition and the information to perform the composition:
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
        alignedin = input.__array_interface__['data'][0] % MEMORY_ALIGNMENT == 0
        alignedout = output.__array_interface__['data'][0] % MEMORY_ALIGNMENT==0
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
            raise ValueError("The composition of an unconstrained input shape o"
                             "perator by an unconstrained output shape operator"
                             " is ambiguous.")
        dtypes = self._get_dtypes(input.dtype)
        sizes = [product(s) * d.itemsize for s, d in izip(shapes, dtypes)]

        ninplaces, alignments, contiguouss = self._get_requirements()

        # make last operand out-of-place
        if preserve_input and self.operands[-1].flags.inplace or \
           not alignedin and alignments[-1] > 1 or \
           not contiguousin and contiguouss[-1]:
            assert ninplaces[-1] > 0
            ninplaces[-1] -= 1
            ninplaces += [0]
            alignments += [MEMORY_ALIGNMENT if alignedin else 1]
            contiguouss += [contiguousin]

        # make first operand out-of-place
        if sizes[0] < max([s for s in sizes[:ninplaces[0]+1]]) or \
           not alignedout and alignments[0] > 1 or \
           not contiguousout and contiguouss[0]:
            assert ninplaces[0] > 0
            ninplaces[0] -= 1
            ninplaces.insert(0, 0)
            alignments.insert(0, MEMORY_ALIGNMENT if alignedout else 1)
            contiguouss.insert(0, contiguousout)

        bufsizes = self._get_bufsizes(sizes, ninplaces)

        v = shapes, dtypes, ninplaces, bufsizes, alignments, contiguouss 
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
        alignments = []
        contiguouss = []
        ninplaces = []
        ninplace = 0
        alignment = 1
        contiguity = False
        iop = len(self.operands) - 1

        # loop over operators
        while iop >= 0:

            # loop over in-place operators
            while iop >= 0:
                op = self.operands[iop]
                iop -= 1
                if not op.flags.inplace:
                    alignment = max(alignment, op.flags.alignment_input)
                    contiguity = max(contiguity, op.flags.contiguous_input)
                    break
                ninplace += 1
                alignment = max(alignment, op.flags.alignment_input)
                contiguity = max(contiguity, op.flags.contiguous_input)

            ninplaces.insert(0, ninplace)
            alignments.insert(0, alignment)
            contiguouss.insert(0, contiguity)

            ninplace = 0
            alignment = op.flags.alignment_output
            contiguity = op.flags.contiguous_output

        if not op.flags.inplace:
            ninplaces.insert(0, ninplace)
            alignments.insert(0, alignment)
            contiguouss.insert(0, contiguity)

        return ninplaces, alignments, contiguouss

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


    # if inplace_composition -> preserve_input = False
    # if preserve_input and last operand is inplace: make it outplace
    # find from left-to-right the first operand that can not have
    # the composition output as its output -> define first temporary
    # find from right-to-left the first operand that can not have
    # the composition input as its input.

    # ex : I1 * I2 * O3 * I4 * I5 out-of-place composition, preserve_input:
    #    B    B    B    C    B    A
    # the first operand that can not have the composition output B as its output
    # is I4. B is in the pool for operands > 4
    # the first operand that can not have the composition input A as its input
    # is I4.

    # ex : I1 * I2 * O3 * I4 * I5 out-of-place composition, not preserve_input:
    #    B    B    B    A    A    A
    # the first operand that can not have the composition output B as its output
    # is I4. B is in the pool for operands > 4
    # the first operand that can not have the composition input A as its input
    # is O3.

    # ex : I1 * I2 * O3 * I4 * I5 in-place composition, preserve_input:
    #    A    A    A    B    B    A
    # the first operand that can not have the composition output A as its output
    # is I4. A is in the pool for operands > 4
    # the first operand that can not have the composition input A as its input
    # is I4.

    # ex : I1 * I2 * O3 * I4 * I5 in-place composition, not preserve_input:
    #    A    A    A    B    A    A
    # the first operand that can not have the composition output A as its output
    # is I4. A is in the pool for operands > 4
    # the first operand that can not have the composition input A as its input
    # is O3.

    @classmethod
    def _get_attributes(cls, operands, **keywords):
        shapes = cls._get_shapes(operands[-1].shapein, operands[0].shapeout,
                                 operands)
        attr = {
            'attrin':cls._merge_attr([o.attrin for o in operands]),
            'attrout':cls._merge_attr([o.attrout for o in operands[::-1]]),
            'classin':first_is_not((o.classin for o in operands[::-1]), None),
            'classout':first_is_not((o.classout for o in operands), None),
            'commin':first_is_not((o.commin for o in operands[::-1]), None),
            'commout':first_is_not((o.commout for o in operands), None),
            'dtype':cls._find_common_type(o.dtype for o in operands),
            'flags':cls._merge_flags(operands),
            'shapein':shapes[-1],
            'shapeout':shapes[0],
            'reshapein':cls._merge_reshapein(operands),
            'reshapeout':cls._merge_reshapeout(operands),
            'toshapein':operands[-1].toshapein,
            'toshapeout':operands[0].toshapeout,
            'validatein':operands[-1].validatein,
            'validateout':operands[0].validateout,
        }
        for k, v in keywords.items():
            if k is not 'flags':
                attr[k] = v
        attr['flags'].update(Operator.validate_flags(keywords.get('flags', {})))
        return attr

    @classmethod
    def _merge(cls, op, op1, op2):
        """
        Ensure that op = op1*op2 has a correct shapein, shapeout, etc.

        """
        # bail if the merging has already been done
        if any(isinstance(o, CompositionOperator) for o in [op1, op2]):
            return

        keywords = cls._get_attributes([op1, op2])

        # The merged operator is not guaranteed to handle inplace reductions
        del keywords['flags']['inplace_reduction']

        # reset attributes
        for attr in OPERATOR_ATTRIBUTES + ['_C', '_T', '_H', '_I']:
            if attr in op.__dict__ and attr != 'flags':
                del op.__dict__[attr]

        # re-init operator with merged attributes
        Operator.__init__(op, **keywords)

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
            'linear':all(op.flags.linear for op in operands),
            'real':all(op.flags.real for op in operands),
            'square':all(op.flags.square for op in operands),
            'separable':all(op.flags.separable for op in operands),
            'inplace_reduction':operands[0].flags.inplace_reduction,
            'alignment_input':operands[-1].flags.alignment_input,
            'alignment_output':operands[0].flags.alignment_output,
            'contiguous_input':operands[-1].flags.contiguous_input,
            'contiguous_output':operands[0].flags.contiguous_output,
}

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

        dtype = self._find_common_type(o.dtype for o in operands)
        switch_T_H = self.flags.real and dtype is not None and dtype.kind == 'c'
        if switch_T_H:
            T, H, IT, IH = '.H', '.T', '.IH', '.IT'
        else:
            T, H, IT, IH = '.T', '.H', '.IT', '.IH'

        self.set_rule('.C', lambda s:DirectOperatorFactory(GroupOperator, s,
            [m.C for m in s.operands], name=self.__name__ + '.C'))
        self.set_rule(T, lambda s:ReverseOperatorFactory(GroupOperator, s,
            [m.T for m in s.operands[::-1]], name=self.__name__ + '.T'))
        self.set_rule(H, lambda s:ReverseOperatorFactory(GroupOperator, s,
            [m.H for m in s.operands[::-1]], name=self.__name__ + '.H'))
        self.set_rule('.I', lambda s:ReverseOperatorFactory(GroupOperator, s,
            [m.I for m in s.operands[::-1]], name=self.__name__ + '.I'))
        self.set_rule('.IC', lambda s:ReverseOperatorFactory(GroupOperator, s,
            [m.I.C for m in s.operands[::-1]], name=self.__name__ + '.I.C'))
        self.set_rule(IT, lambda s:DirectOperatorFactory(GroupOperator, s,
            [m.I.T for m in s.operands], name=self.__name__ + '.I.T'))
        self.set_rule(IH, lambda s:DirectOperatorFactory(GroupOperator, s,
            [m.I.H for m in s.operands], name=self.__name__ + '.I.H'))
        self.del_rule('.{self}', CompositionOperator)
        self.del_rule('.{Operator}', CompositionOperator)
        self.del_rule('{Operator}.', CompositionOperator)


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
            self.__class__ = operands[0].__class__
            self.__dict__ = operands[0].__dict__.copy()
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
                raise ValueError('If the block operator input shape has one mor'
                                 'e dimension than its blocks, the input partit'
                                 'ion must be a tuple of ones.')
        if new_axisout is not None:
            if partitionout is None:
                partitionout = len(operands) * (1,)
            elif any(p not in (None, 1) for p in partitionout):
                raise ValueError('If the block operator output shape has one mo'
                                 're dimension than its blocks, the output part'
                                 'ition must be a tuple of ones.')

        if axisin is not None and new_axisin is not None:
            raise ValueError("The keywords 'axisin' and 'new_axisin' are exclus"
                             "ive.")
        if axisout is not None and new_axisout is not None:
            raise ValueError("The keywords 'axisout' and 'new_axisout' are excl"
                             "usive.")

        if partitionin is partitionout is None:
            raise ValueError('No partition is provided.')
        if partitionin is not None:
            if len(partitionin) != len(operands):
                raise ValueError('The number of operators must be the same as t'
                                 'he length of the input partition.')
            partitionin = merge_none(partitionin, self._get_partitionin(
                                      operands, partitionout, axisin, axisout,
                                      new_axisin, new_axisout))
        if partitionout is not None:
            if len(partitionout) != len(operands):
                raise ValueError('The number of operators must be the same as t'
                                 'he length of the output partition.')
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

        self.set_rule('.C', lambda s: BlockOperator([op.C for op in s.operands],
                      s.partitionin, s.partitionout, s.axisin, s.axisout,
                      s.new_axisin, s.new_axisout))
        self.set_rule('.T', lambda s: BlockOperator([op.T for op in s.operands],
                      s.partitionout, s.partitionin, s.axisout, s.axisin,
                      s.new_axisout, s.new_axisin))
        self.set_rule('.H', lambda s: BlockOperator([op.H for op in s.operands],
                      s.partitionout, s.partitionin, s.axisout, s.axisin,
                      s.new_axisout, s.new_axisin))

        if isinstance(self, BlockDiagonalOperator):
            self.set_rule('.I', lambda s: type(s)([op.I for op in
                      s.operands], s.partitionout, s.axisout, s.axisin,
                      s.new_axisout, s.new_axisin))
            self.set_rule('.IC', lambda s: type(s)([op.I.C for op in \
                      s.operands], s.partitionout, s.axisout, s.axisin,
                      s.new_axisout, s.new_axisin))
            self.set_rule('.IT', lambda s: type(s)([op.I.T for op in \
                      s.operands], s.partitionin, s.axisin, s.axisout,
                      s.new_axisin, s.new_axisout))
            self.set_rule('.IH', lambda s: type(s)([o.I.H for o in \
                      s.operands], s.partitionin, s.axisin, s.axisout,
                      s.new_axisin, s.new_axisout))

        self.set_rule('.{Operator}', self._rule_add_operator, AdditionOperator)
        self.set_rule('.{Operator}', self._rule_left_operator,
                      CompositionOperator)
        self.set_rule('{Operator}.', self._rule_right_operator,
                      CompositionOperator)
        self.set_rule('{self}.', self._rule_add_blockoperator, AdditionOperator)
        self.set_rule('{self}.', self._rule_mul_blockoperator,
                      MultiplicationOperator)
        self.set_rule('{BlockOperator}.', self._rule_comp_blockoperator,
                      CompositionOperator)

    def toshapein(self, v):
        if self.shapein is not None:
            return v.reshape(self.shapein)
        if self.partitionin is None:
            return self.operands[0].toshapein(v)
        axisin = self.axisin if self.axisin is not None else self.new_axisin
        if None in self.partitionin or axisin not in (0,-1):
            raise ValueError('Ambiguous reshaping.')
        p = sum(self.partitionin)
        if v.size == p:
            return v
        if axisin == 0:
            return v.reshape((p,-1))
        return v.reshape((-1,p))
            
    def toshapeout(self, v):
        if self.shapeout is not None:
            return v.reshape(self.shapeout)
        if self.partitionout is None:
            return self.operands[0].toshapeout(v)
        axisout = self.axisout if self.axisout is not None else self.new_axisout
        if None in self.partitionout or axisout not in (0,-1):
            raise ValueError('Ambiguous reshaping.')
        p = sum(self.partitionout)
        if v.size == p:
            return v
        if axisout == 0:
            return v.reshape((p,-1))
        return v.reshape((-1,p))

    def _get_attributes(self, operands, **keywords):
        # UGLY HACK: required by self.reshapein/out. It may be better to make
        # the _get_attributes a class method, pass all partitionin/out etc
        # stuff and inline the reshapein/out methods to get shapein/shapeout.
        self.operands = operands

        attr = {
            'attrin':first_is_not((o.attrin for o in operands), None),
            'attrout':first_is_not((o.attrout for o in operands), None),
            'classin':first_is_not((o.classin for o in operands), None),
            'classout':first_is_not((o.classout for o in operands), None),
            'commin':first_is_not((o.commin for o in operands), None),
            'commout':first_is_not((o.commout for o in operands), None),
            'dtype':self._find_common_type(o.dtype for o in operands),
            'flags':self._merge_flags(operands),
            'shapein':self.reshapeout(None),
            'shapeout':self.reshapein(None),
        }
        for k, v in keywords.items():
            if k is not 'flags':
                attr[k] = v
        attr['flags'].update(Operator.validate_flags(keywords.get('flags', {})))
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
            raise ValueError("The blocks do not have the same number of dimensi"
                             "ons: '{0}'.".format(shapes))
        if any(shapes[i] is not None and shapes[i][axis] != p[i]
                for i in range(len(p)) if p[i] is not None):
            raise ValueError("The blocks have shapes '{0}' incompatible with th"
                             "e partition {1}.".format(shapes, p))
        if len(explicit) != 1:
            ok = [all(s is None or s[i] == shape[i] for s in shapes)
                  for i in range(rank)]
            ok[axis] = True
            if not all(ok):
                raise ValueError("The dimensions of the blocks '{0}' are not th"
                                 "e same along axes other than that of the part"
                                 "ition '{1}'.".format(shapes, p))

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
        return {'linear':all(op.flags.linear for op in operands),
                'real':all(op.flags.real for op in operands),
                'inplace_reduction':False}

    def reshapein(self, shapein):
        shapeins = self._get_shape_operands(shapein, self.partitionin,
            self.partitionout, self.axisin, self.new_axisin)
        shapeouts = [o.shapeout if s is None else tointtuple(o.reshapein(s))
                     for o, s in zip(self.operands, shapeins)]
        return self._get_shape_composite(shapeouts, self.partitionout,
                                         self.axisout, self.new_axisout)

    def reshapeout(self, shapeout):
        shapeouts = self._get_shape_operands(shapeout, self.partitionout,
            self.partitionin, self.axisout, self.new_axisout)
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
        axisin1= first_is_not([op1.axisin, op1.new_axisin], None)
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
    def _rule_add_operator(self, op):
        """ Rule for BlockOperator + Operator. """
        if not op.flags.separable:
            return None
        return BlockOperator([o + op for o in self.operands],
            self.partitionin, self.partitionout, self.axisin, self.axisout,
            self.new_axisin, self.new_axisout)

    @staticmethod
    def _rule_right_operator(op, self):
        """ Rule for Operator * BlockOperator. """
        if self.partitionout is None:
            return None
        if isinstance(op, BlockOperator):
            return None
        if not op.flags.separable:
            return None
        n = len(self.partitionout)
        partitionout = self._get_partitionout(n*[op], self.partitionout,
            self.axisout, self.axisout, self.new_axisout, self.new_axisout)
        return BlockOperator([op * o for o in self.operands],
            self.partitionin, partitionout, self.axisin, self.axisout,
            self.new_axisin, self.new_axisout)

    @staticmethod
    def _rule_left_operator(self, op):
        """ Rule for BlockOperator * Operator. """
        if self.partitionin is None:
            return None
        if not op.flags.separable:
            return None
        n = len(self.partitionin)
        partitionin = self._get_partitionin(n*[op], self.partitionin,
            self.axisin, self.axisin, self.new_axisin, self.new_axisin)
        return BlockOperator([o * op for o in self.operands],
            partitionin, self.partitionout, self.axisin, self.axisout,
            self.new_axisin, self.new_axisout)

    @staticmethod
    def _rule_commutative_blockoperator(p1, p2, operation):
        """ Rule for BlockOperator + BlockOperator. """
        partitions = p1._validate_partition_commutative(p1, p2)
        if partitions is None:
            return None
        partitionout, partitionin = partitions
        operands = [operation([o1, o2]) for o1,o2 in \
                    zip(p1.operands, p2.operands)]
        return BlockOperator(operands, partitionin, partitionout,
            p1.axisin, p1.axisout, p1.new_axisin, p1.new_axisout)

    @staticmethod
    def _rule_add_blockoperator(p1, p2):
        return p1._rule_commutative_blockoperator(p1, p2, AdditionOperator)

    @staticmethod
    def _rule_mul_blockoperator(p1, p2):
        return p1._rule_commutative_blockoperator(p1, p2, MultiplicationOperator)

    @staticmethod
    def _rule_comp_blockoperator(p1, p2):
        """ Rule for BlockOperator * BlockOperator. """
        partitions = p1._validate_partition_composition(p1, p2)
        if partitions is None:
            return None
        partitionout, partitionin = partitions
        operands = [o1 * o2 for o1,o2 in zip(p1.operands, p2.operands)]
        if partitionin is partitionout is None:
            return AdditionOperator(operands)
        axisin, axisout = p2.axisin, p1.axisout
        new_axisin, new_axisout = p2.new_axisin, p1.new_axisout
        return BlockOperator(operands, partitionin, partitionout, axisin,
            axisout, new_axisin, new_axisout)


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

        if axisin is None and new_axisin is None:
            raise NotImplementedError('Free partitioning not implemented yet.')

        if axisout is None:
            axisout = axisin
        if new_axisout is None:
            new_axisout = new_axisin

        if partitionin is None:
            partitionin = self._get_partition([op.shapein \
                for op in operands], axisin, new_axisin)
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
                    raise ValueError('The shape of an operator with implicit pa'
                        'rtition cannot be inferred.')
                shapein = list(input.shape)
                shapein[self.axisin] = self.partitionin[i]
                partitionout[i] = tointtuple(o.reshapein(shapein))[self.axisout]
        else:
            partitionout = self.partitionout

        for op, sin, sout in zip(self.operands, self.get_slicesin(),
                                 self.get_slicesout(partitionout)):
            i = input[sin]
            o = output[sout]
            with _pool.copy_if(i, op.flags.alignment_input,
                               op.flags.contiguous_input) as i:
                with _pool.copy_if(o, op.flags.alignment_output,
                                   op.flags.contiguous_output) as o:
                    op.direct(i, o)

    @staticmethod
    def _merge_flags(operands):
        flags = BlockOperator._merge_flags(operands)
        flags.update({'square':all(op.flags.square for op in operands),
                      'symmetric':all(op.flags.symmetric for op in operands),
                      'hermitian':all(op.flags.hermitian for op in operands),
                      'inplace':all(op.flags.inplace for op in operands)})
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
    >>> op = BlockColumnOperator([I,2*I])
    >>> op.todense()

    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 2.,  0.,  0.],
           [ 0.,  2.,  0.],
           [ 0.,  0.,  2.]])

    """   
    def __init__(self, operands, partitionout=None, axisout=None,
                 new_axisout=None, **keywords):

        operands = self._validate_operands(operands)

        if axisout is None and new_axisout is None:
            raise NotImplementedError('Free partitioning not implemented yet.')

        if partitionout is None:
            partitionout = self._get_partition([op.shapeout \
                for op in operands], axisout, new_axisout)
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
            with _pool.copy_if(o, op.flags.alignment_output,
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
    >>> op = BlockRowOperator([I,2*I])
    >>> op.todense()

    array([[ 1.,  0.,  0., 2., 0., 0.],
           [ 0.,  1.,  0., 0., 2., 0.],
           [ 0.,  0.,  1., 0., 0., 2.]])

    """   
    def __init__(self, operands, partitionin=None, axisin=None, new_axisin=None,
                 operation=operator.iadd, **keywords):

        operands = self._validate_operands(operands)

        if axisin is None and new_axisin is None:
            raise NotImplementedError('Free partitioning not implemented yet.')

        if partitionin is None:
            partitionin = self._get_partition([op.shapein for op in
                operands], axisin, new_axisin)
        partitionin = tointtuple(partitionin)

        keywords['flags'] = Operator.validate_flags(keywords.get('flags', {}),
                                linear=operation is operator.iadd)
        BlockOperator.__init__(self, operands, partitionin=partitionin, axisin=
                               axisin, new_axisin=new_axisin, **keywords)

        self.operation = operation
        self._need_temporary = any(not o.flags.inplace_reduction for o in
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
        with _pool.copy_if(i, op.flags.alignment_input,
                           op.flags.contiguous_input) as i:
            op.direct(i, output)

        with _pool.get_if(self._need_temporary, output.shape, output.dtype,
                          self.__name__) as buf:

            for op, sin in zip(self.operands, sins)[1:]:
                i = input[sin]
                with _pool.copy_if(i, op.flags.alignment_input,
                                   op.flags.contiguous_input) as i:
                    if op.flags.inplace_reduction:
                        op.direct(i, output, operation=self.operation)
                    else:
                        op.direct(i, buf)
                        self.operation(output, buf)

    def __str__(self):
        operands = [str(o) for o in self.operands]
        if len(operands) > 2:
            operands = [operands[0], '...', operands[-1]]
        return '[[ ' + ' '.join(operands) + ' ]]'


@linear
@real
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
        self.set_rule('.T', lambda s: ReverseOperatorFactory(ReshapeOperator,s))
        self.set_rule('.T.', '1', CompositionOperator)

    def direct(self, input, output):
        if isalias(input, output):
            pass
        output.ravel()[:] = input.ravel()

    def __str__(self):
        return strshape(self.shapeout) + '←' + strshape(self.shapein)


class BroadcastingOperator(Operator):
    """
    Abstract class for operators that operate on a data array and
    the input array, and for which broadcasting of the data array across
    the input array is required.

    Leftward broadcasting is the normal numpy's broadcasting along the slow
    dimensions, if the array is stored in C order. Rightward broadcasting is
    a broadcasting along the fast dimension.
    """
    def __init__(self, data, broadcast=None, shapeout=None, **keywords):
        if data is None:
            raise ValueError('The input data is None.')
        if 'dtype' in keywords:
            dtype = keywords['dtype']
        else:
            data = np.asarray(data)
            dtype = keywords['dtype'] = data.dtype
        data = np.array(data, dtype, order='c', copy=False)
        if broadcast is None:
            broadcast = 'scalar' if data.ndim == 0 else 'disabled'
        else:
            broadcast = broadcast.lower()
        values = ('leftward', 'rightward', 'disabled', 'scalar')
        if broadcast not in values:
            raise ValueError("Invalid value '{0}' for the broadcast keyword. Ex"
                "pected values are {1}.".format(broadcast, strenum(values)))
        if broadcast == 'disabled':
            if shapeout not in (None, data.shape):
                raise ValueError("The input shapein is incompatible with the da"
                                 "ta shape.")
            shapeout = data.shape
        self.broadcast = broadcast
        self.data = data
        Operator.__init__(self, shapeout=shapeout, **keywords)
        self.set_rule('{BroadcastingOperator}.', lambda b1, b2: \
            self._rule_broadcast(b1, b2, np.add), AdditionOperator)
        self.set_rule('{BroadcastingOperator}.', lambda b1, b2: \
            self._rule_broadcast(b1, b2, np.multiply), CompositionOperator)
        self.set_rule('.{BlockOperator}', lambda s,o: self._rule_left_block(s,
                      o, CompositionOperator), CompositionOperator)
        self.set_rule('{BlockOperator}.', lambda o,s: self._rule_right_block(o,
                      s, CompositionOperator), CompositionOperator)
        self.set_rule('.{BlockOperator}', lambda s,o: self._rule_left_block(s,
                      o, AdditionOperator), AdditionOperator)
        self.set_rule('.{BlockOperator}', lambda s,o: self._rule_left_block(s,
                      o, MultiplicationOperator), MultiplicationOperator)

    def get_data(self):
        return self.data

    @staticmethod
    def _rule_broadcast(b1, b2, operation):
        # this rule only returns direct subclasses of BroadcastingOperator:
        i1 = b1.__class__.__mro__.index(BroadcastingOperator) - 1
        try:
            i2 = b2.__class__.__mro__.index(BroadcastingOperator) - 1
        except ValueError:
            i2 = -1
        if i1 == i2 == -1:
            cls = BroadcastingOperator
        elif i1 == -1:
            cls = b2.__class__.__mro__[i2]
        elif i2 == -1:
            cls = b1.__class__.__mro__[i1]
        else:
            cls = b1.__class__.__mro__[i1]
            if cls is not b2.__class__.__mro__[i2]:
                return None

        # check broadcast
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
            if new_axis is not None and ndim == 1 and (new_axis == -1 and 
               b == 'rightward' or new_axis ==  0 and b == 'leftward'):
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
            nargs = len(argspec.args) - (len(argspec.defaults)
                    if argspec.defaults is not None else 0) - 1
            slices = op._get_slices(partition, axis, new_axis)
            ops = []
            for s, o in zip(slices, op.operands):
                if nargs == 0:
                    replicated = type(self)(shapeout=data[s].shape, *args,
                                            **keywords)
                else:
                    replicated = type(self)(data[s], broadcast=b, *args,
                                            **keywords)
                ops.append(func_operation(replicated, o))

        return BlockOperator(ops, op.partitionin, op.partitionout, op.axisin,
                             op.axisout, op.new_axisin, op.new_axisout)

    @staticmethod
    def _rule_left_block(self, op, cls):
        func_op = lambda d, b: cls([d, b])
        return self._rule_block(self, op, op.shapeout, op.partitionout,
                                op.axisout, op.new_axisout, func_op)

    @staticmethod
    def _rule_right_block(op, self, cls):
        func_op = lambda d, b: cls([b, d])
        return self._rule_block(self, op, op.shapein, op.partitionin,
                                op.axisin, op.new_axisin, func_op)

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


@symmetric
@inplace
class DiagonalOperator(BroadcastingOperator):
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
    >>> A = DiagonalOperator(arange(1, 6, 2))
    >>> A.todense()

    array([[1, 0, 0],
           [0, 3, 0],
           [0, 0, 5]])

    >>> A = DiagonalOperator(arange(1, 3), broadcast='rightward', shapein=(2, 2))
    >>> A.todense()

    array([[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 2, 0],
           [0, 0, 0, 2]])

    """
    def __init__(self, data, broadcast=None, **keywords):
        data = np.asarray(data)
        if broadcast is None:
            broadcast = 'scalar' if data.ndim == 0 else 'disabled'
        if broadcast == 'disabled':
            keywords['shapein'] = data.shape
            keywords['shapeout'] = data.shape
        n = data.size
        nmones, nzeros, nones, other, same = inspect_special_values(data)
        if 'dtype' not in keywords and nzeros + nones == n:
            keywords['dtype'] = None
        if nzeros == n and not isinstance(self, ZeroOperator):
            self.__class__ = ZeroOperator
            self.__init__(**keywords)
            return
        if nones == n and not isinstance(self, IdentityOperator):
            self.__class__ = IdentityOperator
            self.__init__(**keywords)
            return
        if same and not isinstance(self, (HomothetyOperator, ZeroOperator)):
            self.__class__ = HomothetyOperator
            self.__init__(data.flat[0], **keywords)
            return
        if nones + nzeros == n and not isinstance(self, (HomothetyOperator,
                                                         MaskOperator)):
            self.__class__ = MaskOperator
            self.__init__(~data.astype(np.bool8), **keywords)
            return
        if nmones + nones == n:
            keywords['flags'] = self.validate_flags(keywords.get('flags', {}),
                                                    involutary=True)
        BroadcastingOperator.__init__(self, data, broadcast, **keywords)

    def direct(self, input, output):
        if self.broadcast == 'rightward':
            np.multiply(input.T, self.get_data().T, output.T)
        else:
            np.multiply(input, self.get_data(), output)

    def conjugate_(self, input, output):
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

    def validatein(self, shape):
        if self.data.size == 1:
            return
        n = self.data.ndim
        if len(shape) < n:
            raise ValueError("Invalid number of dimensions.")
        
        if self.broadcast == 'rightward':
            it = zip(shape[:n], self.data.shape[:n])
        else:
            it = zip(shape[-n:], self.data.shape[-n:])
        for si, sd in it:
            if sd != 1 and sd != si:
                raise ValueError("The data array cannot be broadcast across the"
                                 " input.")

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


@real
@idempotent
class MaskOperator(DiagonalOperator):
    """
    A subclass of DiagonalOperator with 0 (True) and 1 (False) on the diagonal.

    Exemple
    -------
    >>> M = MaskOperator([True, False])
    >>> M.todense()

    array([[False, False],
           [False,  True]], dtype=bool)

    Notes
    -----
    We follow the convention of MaskedArray, where True means masked.

    """
    def __init__(self, mask, **keywords):
        mask = np.array(mask, dtype=np.bool8, copy=False)
        nmones, nzeros, nones, other, same = inspect_special_values(mask)
        if nzeros == mask.size:
            self.__class__ = IdentityOperator
            self.__init__(**keywords)
            return
        if nones == mask.size:
            self.__class__ = ZeroOperator
            self.__init__(**keywords)
            return
        BroadcastingOperator.__init__(self, mask, **keywords)

    def direct(self, input, output):
        ufuncs.masking(input, self.data, output)

    def get_data(self):
        return ~self.data


@inplace
class HomothetyOperator(DiagonalOperator):
    """
    Multiplication by a scalar.

    """
    def __init__(self, data, **keywords):
        if 'broadcast' in keywords:
            if keywords['broadcast'] != 'scalar':
                raise ValueError("Invalid broadcast value '{0}'.".format(
                                 keywords['broadcast']))
            del keywords['broadcast']
        data = np.asarray(data)
        if data.ndim > 0:
            if any(s != 0 for s in data.strides) and \
               np.any(data.flat[0] != data):
                raise ValueError("The input is not a scalar..")
            data = np.asarray(data.flat[0])

        DiagonalOperator.__init__(self, data, 'scalar', **keywords)
        self.set_rule('.C', lambda s: DirectOperatorFactory(HomothetyOperator,
                      s, np.conjugate(s.data)))
        self.set_rule('.I', lambda s: DirectOperatorFactory(HomothetyOperator,
                      s, 1/s.data if s.data != 0 else np.nan))
        self.set_rule('.IC', lambda s: DirectOperatorFactory(HomothetyOperator,
                      s, np.conjugate(1/s.data) if s.data != 0 else np.nan))
        self.set_rule('{Operator}.', self._rule_right, CompositionOperator)

    def __str__(self):
        data = self.data.flat[0]
        if data == int(data):
            data = int(data)
        if data == 1:
            return 'I'
        if data == -1:
            return '-I'
        return str(data) + 'I'

    @staticmethod
    def _rule_right(operator, self):
        if operator.flags.linear:
            return self, operator


@real
@idempotent
@involutary
@inplace
class IdentityOperator(HomothetyOperator):
    """
    A subclass of HomothetyOperator with data = 1.

    Exemple
    -------
    >>> I = IdentityOperator()
    >>> I.todense(3)

    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    >>> I = IdentityOperator(shapein=2)
    >>> I * arange(2)
    ndarraywrap([ 0.,  1.])

    """
    def __init__(self, shapein=None, **keywords):
        HomothetyOperator.__init__(self, 1, shapein=shapein, **keywords)
        self.set_rule('.{Operator}', self._rule_left, CompositionOperator)
        self.set_rule('.{Operator}', lambda s,o: o, MultiplicationOperator)

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
class ConstantOperator(BroadcastingOperator):
    """
    Non-linear constant operator.

    """
    def __init__(self, data, broadcast=None, **keywords):
        data = np.asarray(data)
        if broadcast is None:
            broadcast = 'scalar' if data.ndim == 0 else 'disabled'
        if broadcast == 'disabled':
            keywords['shapeout'] = data.shape
        if data.ndim > 0 and np.all(data == data.flat[0]):
            self.__init__(data.flat[0], **keywords)
            return
        if not isinstance(self, ZeroOperator) and data.ndim == 0  and data == 0:
            self.__class__ = ZeroOperator
            self.__init__(**keywords)
            return
        BroadcastingOperator.__init__(self, data, broadcast, **keywords)
        self.set_rule('.C', lambda s: DirectOperatorFactory(ConstantOperator, s,
                      s.data.conjugate(), broadcast=s.broadcast))
#        if self.flags.shape_input == 'unconstrained' and \
#           self.flags.shape_output != 'implicit':
#            self.set_rule('.T', '.')
        self.set_rule('.{Operator}', self._rule_left, CompositionOperator)
        self.set_rule('{Operator}.', self._rule_right, CompositionOperator)
        self.set_rule('.{CompositionOperator}', self._rule_mul,
                      MultiplicationOperator)
        self.set_rule('.{DiagonalOperator}', self._rule_mul,
                      MultiplicationOperator)
        self.set_rule('.{BlockOperator}', lambda s,o: self.
                      _rule_left_block_composition(s, o), CompositionOperator)
        self.set_rule('{BlockOperator}.', lambda o,s: self.
                      _rule_right_block_composition(o, s), CompositionOperator)

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
        return CompositionOperator([DiagonalOperator(self.data, broadcast=
            self.broadcast, shapein=self.shapeout), op])

    @staticmethod
    def _rule_left_block_composition(self, op):
        if self.flags.shape_output != 'explicit':
            return
        if self.shapeout != op.shapeout:
            return
        func_op = lambda c, b: CompositionOperator([c, b])
        return self._rule_block(self, op, op.shapeout, op.partitionout,
                                op.axisout, op.new_axisout, func_op)

    @staticmethod
    def _rule_right_block_composition(op, self):
        return

    def __str__(self):
        return str(self.data)

    def __neg__(self):
        return ConstantOperator(-self.data, broadcast=self.broadcast,
            shapein=self.shapein, shapeout=self.shapeout,
            reshapein=self.reshapein, reshapeout=self.reshapeout,
            dtype=self.dtype)


@linear
@real
class ZeroOperator(ConstantOperator):
    """
    A subclass of ConstantOperator with data = 0.
    """
    def __init__(self, *args, **keywords):
        ConstantOperator.__init__(self, 0, **keywords)
        self.set_rule('.T', lambda s: ReverseOperatorFactory(ZeroOperator, s))
        self.set_rule('.{Operator}', lambda s,o: o, AdditionOperator)

    def direct(self, input, output, operation=operation_assignment):
        operation(output, 0)

    @staticmethod
    def _rule_right(op, self):
        if op.commin is not None or op.commout is not None:
            return None
        if op.flags.linear:
            return self.copy()
        return super(ZeroOperator, self)._rule_right(op, self)

    def __neg__(self):
        return self


@linear
class DenseOperator(Operator):
    """
    Operator representing a dense matrix.

    Example
    -------
    >>> m = array([[1.,2.,3.],[2.,3.,4.]])
    >>> d([1,0,0])
    array([ 1.,  2.])
    """
    def __init__(self, data, shapein=None, shapeout=None, dtype=None,
                 **keywords):
        if data is None:
            raise ValueError('The input data is None.')
        data = np.asarray(data)
        if dtype is None:
            dtype = data.dtype
        data = np.array(data, dtype, copy=False)
        if data.ndim != 2:
            raise ValueError('The input is not a 2-dimensional array.')
        if shapein is None:
            shapein = data.shape[1]
        elif np.product(shapein) != data.shape[1]:
            raise ValueError("The input shape '{0}' is incompatible with that o"
                             "f the input matrix '{1}'.".format(strshape(
                             shapein), data.shape[1]))
        if shapeout is None:
            shapeout = data.shape[0]
        elif np.product(shapeout) != data.shape[0]:
            raise ValueError("The input shape '{0}' is incompatible with that o"
                             "f the input matrix '{1}'.".format(strshape(
                             shapeout), data.shape[0]))
        Operator.__init__(self, shapein=shapein, shapeout=shapeout, dtype=dtype,
                          **keywords)
        self.data = data
        self.set_rule('.C', lambda s: DirectOperatorFactory(type(s), s,
            np.conjugate(s.data)))
        self.set_rule('.T', lambda s: ReverseOperatorFactory(type(s), s,
            s.data.T))
        self.set_rule('.H', lambda s: ReverseOperatorFactory(type(s), s,
            np.conjugate(s.data.T)))

    def direct(self, input, output):
        np.dot(self.data, input.ravel(), output)

    def todense(self, shapein=None):
        return self.data


@real
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
                raise TypeError("The input ufunc '{0}' has {1} input argument. "
                                "Expected number is 2.".format(
                                func.__name__, func.nin))
            if func.nout != 1:
                raise TypeError("The input ufunc '{0}' has {1} output arguments"
                                ". Expected number is 1.".format(
                                func.__name__, func.nout))
            if np.__version__ < '1.8':
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
                raise TypeError("The input function '{0}' does not have an 'axi"
                                "s' argument.".format(func.__name__))
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
        if len(shape) < (self.axis+1 if self.axis>=0 else abs(self.axis)):
            raise ValueError('The input shape has an insufficient number of dim'
                             'ensions.')


def DirectOperatorFactory(cls, source, *args, **keywords):
    for attr in OPERATOR_ATTRIBUTES:
        if attr in keywords or attr in ['dtype', 'flags']:
            continue
        keywords[attr] = getattr(source, attr)
    keywords['dtype'] = source.dtype
    keywords['flags'] = Operator.validate_flags(keywords.get('flags', {}))
    keywords['flags']['real'] = source.flags.real
    keywords['flags']['square'] = source.flags.square
    keywords['flags']['inplace'] = source.flags.inplace
    keywords['flags']['alignment_input'] = source.flags.alignment_input
    keywords['flags']['alignment_output'] = source.flags.alignment_output
    keywords['flags']['contiguous_input'] = source.flags.contiguous_input
    keywords['flags']['contiguous_output'] = source.flags.contiguous_output
    return cls(*args, **keywords)


def ReverseOperatorFactory(cls, source, *args, **keywords):
    for attr in OPERATOR_ATTRIBUTES:
        if attr in keywords or attr in ['dtype', 'flags']:
            continue
        if attr == 'reshapein' and source.reshapeout == \
           Operator.reshapeout.__get__(source, type(source)):
            continue
        if attr == 'reshapeout' and source.reshapein == \
           Operator.reshapein.__get__(source, type(source)):
            continue
        if attr.endswith('in'):
            attr_source = attr[:-2] + 'out'
        elif attr.endswith('out'):
            attr_source = attr[:-3] + 'in'
        else:
            attr_source = attr
        keywords[attr] = getattr(source, attr_source)
    keywords['dtype'] = source.dtype
    keywords['flags'] = Operator.validate_flags(keywords.get('flags', {}))
    keywords['flags']['real'] = source.flags.real
    keywords['flags']['square'] = source.flags.square
    keywords['flags']['inplace'] = source.flags.inplace
    keywords['flags']['alignment_input'] = source.flags.alignment_output
    keywords['flags']['alignment_output'] = source.flags.alignment_input
    keywords['flags']['contiguous_input'] = source.flags.contiguous_output
    keywords['flags']['contiguous_output'] = source.flags.contiguous_input
    return cls(*args, **keywords)


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
    
    if isinstance(x, np.ufunc):
        return Operator(x, **keywords)

    if callable(x):
        def direct(input, output):
            output[...] = x(input)
        keywords['flags'] = Operator.validate_flags(keywords.get('flags', {}),
                                                    inplace=True)
        return Operator(direct, **keywords)

    if isinstance(x, (list, tuple)) and len(x) > 0 and \
       isinstance(x[0], (list, tuple)):
        x = np.array(x)

    if constant and isinstance(x, (int, float, complex, np.bool_, np.number,
                               np.ndarray)) and not isinstance(x, np.matrix):
        return ConstantOperator(x, **keywords)

    if isinstance(x, (np.matrix, np.ndarray)):
        if x.ndim > 0:
            return DenseOperator(x, **keywords)
        x = x[()]
            
    if isinstance(x, (int, float, complex, np.bool_, np.number)):
        return HomothetyOperator(x, **keywords)

    return asoperator(scipy.sparse.linalg.aslinearoperator(x), **keywords)


def asoperator1d(x):
    x = asoperator(x)
    r = ReshapeOperator(x.shape[1], x.shapein)
    s = ReshapeOperator(x.shapeout, x.shape[0])
    return s * x * r

I = IdentityOperator()
O = ZeroOperator()

_pool = MemoryPool()

