#coding: utf-8
"""
The core module defines the Operator class. Operators are functions
which can be added, composed or multiplied by a scalar. See the
Operator docstring for more information.
"""

from __future__ import division

import copy
import gc
import inspect
import numpy as np
import scipy.sparse.linalg
import types

from collections import namedtuple
from .utils import isscalar, tointtuple, strenum
from .decorators import (real, idempotent, involutary, orthogonal, square,
                         symmetric)

__all__ = [
    'Operator',
    'OperatorFlags',
    'AdditionOperator',
    'BroadcastingOperator',
    'CompositionOperator',
    'ConcatenationOperator',
    'IdentityOperator',
    'PartitionOperator',
    'ReshapeOperator',
    'ScalarOperator',
    'asoperator',
]

verbose = True

class OperatorFlags(namedtuple('OperatorFlags',
                               ['LINEAR',
                                'SQUARE',     # shapein == shapeout
                                'REAL',       # o.C = o
                                'SYMMETRIC',  # o.T = o
                                'HERMITIAN',  # o.H = o
                                'IDEMPOTENT', # o * o = o
                                'ORTHOGONAL', # o * o.T = I
                                'UNITARY',    # o * o.H = I
                                'INVOLUTARY', # o * o = I
                                ])):
    """Informative flags about the operator."""
    def __str__(self):
        n = max([len(f) for f in self._fields])
        fields = [ '  ' + f.ljust(n) + ' : ' for f in self._fields]
        return '\n'.join([f + str(v) for f,v in zip(fields,self)])

    def __repr__(self):
        n = max([len(f) for f in self._fields])
        fields = [ f.ljust(n) + '= ' for f in self._fields]
        return self.__class__.__name__ + '(\n  ' + ',\n  '.join([f + str(v) \
            for f,v in zip(fields,self)]) + ')'

class OperatorRule(object):
    """ Binary rule on operators.

    A operator rule is a relation that can be expressed by the sentence
    "'subjects' are 'predicate'". An instance of this class, when called with
    two input arguments checks if the inputs are subjects to the rule, and
    returns the predicate if it is the case. Otherwise, it returns None.

    Arguments
    ---------
    operator: Operator
        The reference operator, one of the two subjects

    subject: str
        It defines the relationship between the two subjects that must be
        verified for the rule to apply. It is the concatenation of two
        expressions. One has to be '.' and stands for the reference subject.
        It determines if the relation is reflected (reference operator is
        on the right) or not (reference operator on the left). The other
        expression constrains the other subject, which must be:
            '.' : the reference operator itself.
            '.C' : the conjugate of the reference object
            '.T' : the transpose of the reference object
            '.H' : the adjoint of the reference object
            '{...}' : an instance of the class '...'
        For instance, given a string '.C.', the rule will apply to the inputs
        o1 and o2 if o1 is o2.C. For a condition '.{DiagonalOperator}', the
        rule will apply if o2 is a DiagonalOperator instance.

    predicate: function or str
        If the two objects o1, o2, are subjects of the rule, the predicate
        will be returned. The predicate can also be '1', '.', '.C', '.T' or '.H'
    
    Example
    -------
    >>> rule = OperatorRule('..', '.')
    >>> o = Operator()
    >>> rule(o, o) is o
    True
    >>> rule(o, IdentityOperator()) is None
    True
    
    """
    def __init__(self, operator, subject, predicate):
        if subject[-1] == '.':
            reflected = True
            other = subject[:-1]
        else:
            reflected = False
            other = subject[1:]
        if isinstance(other, str) and other[0] == '1':
            raise ValueError("'1' cannot be a subject.")
        if isinstance(predicate, str) and predicate[0] == '{' and \
           self.predicate[-1] == '}':
            raise ValueError('Predicate cannot be an operator type.')
        self._str_subject = subject
        self._str_predicate = predicate
        self.reflected = reflected
        self.reference = operator
        self.other = other
        self.predicate = predicate

    def __call__(self, other):

        other_ = self._symbol2operator(self.other)
        predicate = self._symbol2operator(self.predicate)

        if isinstance(other_, str):
            if other.__class__.__name__ != other_ and all(b.__name__ != other_\
               for b in other.__class__.__bases__):
                return None
        elif other is not other_:
            return None

        if isinstance(predicate, str) and predicate == '1':
            right = self.reference if self.reflected else other
            return IdentityOperator(right.shapein)
        if isinstance(predicate, Operator):
            return predicate
        if callable(predicate):
            result = predicate(other)
            if isinstance(result, tuple) and len(result) == 1:
                result = result[0]
            return result
        return predicate

    def __eq__(self, other):
        if not isinstance(other, OperatorRule):
            return NotImplemented
        return str(self) == str(other)
    
    def _symbol2operator(self, symbol):
        if not isinstance(symbol, str) or symbol == '1':
            return symbol
        if symbol[0] == '{' and symbol[-1] == '}':
            return symbol[1:-1]
        try:
            return {'.': self.reference,
                    '.C': self.reference.C,
                    '.T': self.reference.T,
                    '.H': self.reference.H,
                    '.I': self.reference.I}[symbol]
        except (KeyError, TypeError):
            raise ValueError('Invalid symbol: {0}'.format(symbol))

    def __str__(self):
        return '{0} = {1}'.format(self._str_subject, self._str_predicate)

    __repr__ = __str__


class Operator(object):
    """Abstract class representing an operator.

    Attributes
    ----------
    shapein : tuple
         operator's input shape.

    shapeout : tuple
         operator's output shape.

    dtype : dtype
         the operator's dtype is used to determine the dtype of its output.
         Unless it is None, the output dtype is the common type of the
         operator and input dtypes. If dtype is None, the output dtype is
         the input dtype.

    C : Operator
         conjugate operator.

    T : Operator
         tranpose operator.

    H : Operator
         adjoint operator.

    I : Operator
         inverse operator.

    """
    def __init__(self, direct=None, transpose=None, adjoint=None,
                 conjugate_=None, inverse=None, inverse_transpose=None,
                 inverse_adjoint=None, inverse_conjugate=None, shapein=None,
                 shapeout=None, reshapein=None, reshapeout=None, dtype=None,
                 flags=None):
            
        for method, name in zip( \
            (direct, transpose, adjoint, conjugate_, inverse, inverse_transpose,
             inverse_adjoint, inverse_conjugate),
            ('direct', 'transpose', 'adjoint', 'conjugate_', 'inverse',
             'inverse_transpose', 'inverse_adjoint', 'inverse_conjugate')):
            if method is not None:
                if not hasattr(method, '__call__'):
                    raise TypeError("The method '%s' is not callable." % name)
                # should also check that the method has at least two arguments
                setattr(self, name, method)

        if self.transpose is None and self.adjoint is not None:
            def transpose(input, output):
                self.adjoint(input.conjugate(), output)
                output[:] = output.conjugate()
            self.transpose = transpose

        if self.adjoint is None and self.transpose is not None:
            def adjoint(input, output):
                self.transpose(input.conjugate(), output)
                output[:] = output.conjugate()

        if self.inverse is None:
            self.inverse_conjugate = None

        self._C = self._T = self._H = self._I = None

        self._set_dtype(dtype)
        self._set_flags(self, flags)
        self._set_rules()
        self._set_name()
        self._set_inout(shapein, shapeout, reshapein, reshapeout)

    shapein = None
    shapeout = None
    dtype = None
    flags = OperatorFlags(*9*(False,))
    _reshapein = None
    _reshapeout = None

    direct = None
    transpose = None
    adjoint = None

    def conjugate_(self, input, output):
        self.direct(input.conjugate(), output)
        output[:] = output.conjugate()

    inverse = None
    inverse_transpose = None
    inverse_adjoint = None

    def inverse_conjugate(self, input, output):
        self.inverse(input.conjugate(), output)
        output[:] = output.conjugate()

    def __call__(self, input, output=None):
        if self.direct is None:
            raise NotImplementedError('Call to ' + self.__name__ + ' is not imp'
                                      'lemented.')
        input, output = self._validate_input(input, output)
        self._propagate(input, output, copy=True)
        self.direct(input, output)
        if type(output) is ndarraywrap and len(output.__dict__) == 0:
            output = output.base
        return output

    @property
    def shape(self):
        shape = (np.product(self.shapeout), np.product(self.shapein))
        if shape[0] is None or shape[1] is None:
            return None
        return shape

    def toshapein(self, v):
        """Reshape a vector into a multi-dimensional array compatible with
        the operator's input shape."""
        if self.shapein is None:
            raise ValueError("The operator '" + self.__name__ + "' does not hav"
                             "e an explicit shape.")
        return v.reshape(self.shapein)

    def toshapeout(self, v):
        """Reshape a vector into a multi-dimensional array compatible with
        the operator's output shape."""
        if self.shapeout is None:
            raise ValueError("The operator '" + self.__name__ + "' does not hav"
                             "e an explicit shape.")
        return v.reshape(self.shapeout)

    @staticmethod
    def same_data(array1, array2):
        return array1.__array_interface__['data'][0] == \
               array2.__array_interface__['data'][0]

    def todense(self, shapein=None):
        """
        Output the dense representation of the Operator
        as a ndarray.

        Arguments
        ---------
        shapein: (default None) None or tuple
          If a shapein is not already associated with the Operator,
          it must me passed to the todense method.
        """
        if not self.flags.LINEAR:
            raise TypeError('The operator is not linear.')
        shapein = shapein or self.shapein
        if shapein is None:
            raise ValueError("The operator has an implicit shape. Use the 'shap"
                             "ein' keyword.")
        shapeout = self.reshapein(shapein)
        m, n = np.product(shapeout), np.product(shapein)
        d = np.empty((n,m), self.dtype).view(ndarraywrap)
        v = np.zeros(n, self.dtype)
        for i in range(n):
            v[i] = 1
            self.direct(v.reshape(shapein), d[i,:].reshape(shapeout))
            v[i] = 0
        if len(d.__dict__) == 0:
            d = d.view(np.ndarray)
        return d.T

    def matvec(self, v, output=None):
        v = self.toshapein(v)
        if output is not None:
            output = self.toshapeout(output)
        input, output = self._validate_input(v, output)
        self.direct(input, output)
        return output.ravel().view(np.ndarray)

    def rmatvec(self, v, output=None):
        return self.T.matvec(v, output)

    def add_rule(self, subject, predicate, operation='composition'):
        """
        Add a rule to the rule list, taking care of duplicates.
        Rules matching classes have a lower priority than the others.
        """
        if operation == 'addition' and subject[-1] == '.':
            subject = '.' + subject[:-1]
        rule = OperatorRule(self, subject, predicate)
        rules = self.rules[operation]
        ids = [r._str_subject for r in rules]
        try:
            index = ids.index(rule._str_subject)
            rules[index] = rule
        except ValueError:
            if rule.other[0] == '{':
                try:
                    index = [r.other[0] for r in rules].index('{')
                    rules.insert(index, rule)
                except ValueError:
                    rules.append(rule)
            else:
                rules.insert(0, rule)

    def associated_operators(self):
        """
        By default, the operators returned by the C, T, H and I properties are
        instanciated from the methods provided in the operator's __init__.
        This method provides a way to override this behavior, by specifying the
        associated operators themselves as values in a dictionary, in which
        items are
            - 'C' : conjugate
            - 'T' : tranpose
            - 'H' : adjoint
            - 'I' : inverse
            - 'IC' : inverse conjugate
            - 'IT' : inverse transpose
            - 'IH' : inverse adjoint
            
        """
        return {}

    @property
    def C(self):
        """Return the complex-conjugate of the operator."""
        if self._C is None:
            self._generate_associated_operators()
        return self._C

    @property
    def T(self):
        """Return the transpose of the operator."""
        if self._T is None:
            self._generate_associated_operators()
        return self._T

    @property
    def H(self):
        """Return the adjoint of the operator."""
        if self._H is None:
            self._generate_associated_operators()
        return self._H

    @property
    def I(self):
        """Return the inverse of the operator."""
        if self._I is None:
            self._generate_associated_operators()
        return self._I

    def conjugate(self):
        """Return the complex-conjugate of the operator. Same as '.C'"""
        return self.C
        
    def _allocate(self, shape, dtype, buf=None):
        """Return an array of given shape and dtype. If a buffer is provided and
        is large enough, it is reused, otherwise a memory allocation takes
        place. Every allocation should go through this method.
        """

        if isscalar(shape):
            shape = (shape,)
        dtype = np.dtype(dtype)

        nbytes = dtype.itemsize * np.product(shape)
        if buf is not None and buf.nbytes <= nbytes:
            if buf.shape == shape and buf.dtype == dtype:
                return self._wrap_ndarray(buf), False
            if isscalar(buf):
                buf = buf.reshape(1)
            buf = buf.view(np.int8).ravel()[:nbytes].view(dtype).reshape(shape)
            return self._wrap_ndarray(buf), False

        if verbose:
            if nbytes < 1024:
                snbytes = str(nbytes) + ' bytes'
            else:
                snbytes = str(nbytes / 2**20) + ' MiB'
            print('Info: Allocating ' + str(shape).replace(' ','') + ' ' + \
                  dtype.type.__name__ + ' = ' + snbytes + ' in ' + \
                  self.__name__ + '.')
        try:
            buf = np.empty(shape, dtype)
        except MemoryError:
            gc.collect()
            buf = np.empty(shape, dtype)

        return self._wrap_ndarray(buf), True

    def _allocate_like(self, a, b):
        """Return an array of same shape and dtype as a given array."""
        return self._allocate(a.shape, a.dtype, b)

    def _wrap_ndarray(self, array):
        """Make an input ndarray an instance of a heap class so that we can
        change its class and attributes."""
        if type(array) is np.ndarray:
            array = array.view(ndarraywrap)
        return array

    def _propagate(self, input, output, copy=False):
        """Set the output's class to that of the input. It also copies input's
        attributes into the output. Note that these changes cannot be propagated
        to a non-subclassed ndarray."""
        output.__class__ = input.__class__
        if copy:
            output.__dict__.update(input.__dict__)
        else:
            output.__dict__ = input.__dict__

    def reshapein(self, shapein):
        """Return operator's output shape."""
        shapein = tointtuple(shapein)
        if None not in (self.shapein, shapein) and self.shapein != shapein:
            raise ValueError("The input shape of {0} is {1}. It is incompatible"
                " with '{2}'.".format(self.__name__, _strshape(self.shapein),
                _strshape(shapein)))
        if self.shapeout is not None:
            return self.shapeout
        if self.flags.SQUARE:
            return shapein
        if self._reshapein is None:
            return None
        return tointtuple(self._reshapein(shapein))

    def reshapeout(self, shapeout):
        """Return operator's input shape."""
        shapeout = tointtuple(shapeout)
        if None not in (self.shapeout, shapeout)  and self.shapeout != shapeout:
            raise ValueError("The output shape of {0} is {1}. It is incompatibl"
                "e with '{2}'.".format(self.__name__, _strshape(self.shapeout),
                _strshape(shapeout)))
        if self.shapein is not None:
            return self.shapein
        if self.flags.SQUARE:
            return shapeout
        if self._reshapeout is None:
            return None
        return tointtuple(self._reshapeout(shapeout))

    @staticmethod
    def _find_common_type(dtypes):
        """Return dtype of greater type rank."""
        dtypes = [d for d in dtypes if d is not None]
        if len(dtypes) == 0:
            return None
        return np.find_common_type(dtypes, [])

    def _generate_associated_operators(self):
        """Compute at once the conjugate, transpose, adjoint and inverse
        operators of the instance and of themselves."""
        names = ('C', 'T', 'H', 'I', 'IC', 'IT', 'IH')
        ops = self.associated_operators()
        if not set(ops.keys()) <= set(names):
            raise ValueError("Invalid associated operators. Expected operators "
                             "are '{0}'".format(','.join(names)))

        if self.flags.REAL:
            C = self
        elif 'C' in ops:
            C = ops['C']
        else:
            C = Operator(self.conjugate_, shapein=self.shapein, shapeout= \
                         self.shapeout, reshapein=self._reshapein, reshapeout= \
                         self._reshapeout, dtype=self.dtype, flags=self.flags)
            C.__name__ = self.__name__ + '.C'

        if self.flags.SYMMETRIC:
            T = self
        elif 'T' in ops:
            T = ops['T']
        else:
            T = Operator(self.transpose, shapein=self.shapeout, shapeout= \
                         self.shapein, reshapein=self._reshapeout, reshapeout=\
                         self._reshapein, dtype=self.dtype, flags=self.flags)
            T.toshapein, T.toshapeout = self.toshapeout, self.toshapein
            T.__name__ = self.__name__ + '.T'

        if self.flags.HERMITIAN:
            H = self
        elif 'H' in ops:
            H = ops['H']
        elif self.flags.REAL:
            H = T
        elif self.flags.SYMMETRIC:
            H = C
        else:
            H = Operator(self.adjoint, shapein=self.shapeout, shapeout= \
                         self.shapein, reshapein=self._reshapeout, reshapeout=\
                         self._reshapein, dtype=self.dtype, flags=self.flags)
            H.toshapein, H.toshapeout = self.toshapeout, self.toshapein
            H.__name__ = self.__name__ + '.H'

        if self.flags.INVOLUTARY:
            I = self
        elif 'I' in ops:
            I = ops['I']
        elif self.flags.ORTHOGONAL:
            I = T
        elif self.flags.UNITARY:
            I = H
        else:
            I = Operator(self.inverse, shapein=self.shapeout, shapeout= \
                         self.shapein, reshapein=self._reshapeout, reshapeout=\
                         self._reshapein, dtype=self.dtype, flags=self.flags)
            I.toshapein, I.toshapeout = self.toshapeout, self.toshapein
            I.__name__ = self.__name__ + '.I'

        if self.flags.REAL:
            IC = I
        elif 'IC' in ops:
            IC = ops['IC']
        elif self.flags.ORTHOGONAL:
            IC = H
        elif self.flags.UNITARY:
            IC = T
        elif self.flags.INVOLUTARY:
            IC = C
        else:
            IC = Operator(self.inverse_conjugate, shapein=self.shapeout,
                          shapeout=self.shapein, reshapein=self._reshapeout,
                          reshapeout=self._reshapein, dtype=self.dtype,
                          flags=self.flags)
            IC.toshapein, IC.toshapeout = self.toshapeout, self.toshapein
            IC.__name__ = self.__name__ + '.I.C'

        if self.flags.ORTHOGONAL:
            IT = self
        elif self.flags.SYMMETRIC:
            IT = I
        elif self.flags.UNITARY:
            IT = C
        elif self.flags.INVOLUTARY:
            IT = T
        elif 'IT' in ops:
            IT = ops['IT']
        else:
            IT = Operator(self.inverse_transpose, shapein=self.shapein,
                          shapeout=self.shapeout, reshapein=self._reshapein,
                          reshapeout=self._reshapeout, dtype=self.dtype,
                          flags=self.flags)
            IT.__name__ = self.__name__ + '.I.T'

        if self.flags.UNITARY:
            IH = self
        elif self.flags.HERMITIAN:
            IH = I
        elif self.flags.ORTHOGONAL:
            IH = C
        elif self.flags.INVOLUTARY:
            IH = H
        elif self.flags.SYMMETRIC:
            IH = IC
        elif self.flags.REAL:
            IH = IT
        elif 'IH' in ops:
            IH = ops['IH']
        else:
            IH = Operator(self.inverse_adjoint, shapein=self.shapein,
                          shapeout=self.shapeout, reshapein=self._reshapein,
                          reshapeout=self._reshapeout, dtype=self.dtype,
                          flags=self.flags)
            IH.__name__ = self.__name__ + '.I.H'

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

    def _set_dtype(self, dtype):
        """A non-complex dtype sets the REAL flag to true"""
        if dtype is not None:
            dtype = np.dtype(dtype)
        self.dtype = dtype
        if self.dtype is None or self.dtype.kind != 'c':
            self.flags = self.flags._replace(REAL=True)

    @staticmethod
    def _set_flags(op, flags):
        """Sets class or instance flags."""
        if flags is not None:
            if isinstance(flags, tuple):
                op.flags = flags
            elif isinstance(flags, dict):
                op.flags = op.flags._replace(**flags)
            else:
                raise ValueError('Invalid input flags.')

        if op.flags.REAL:
            if op.flags.SYMMETRIC:
                op.flags = op.flags._replace(HERMITIAN=True)
            if op.flags.HERMITIAN:
                op.flags = op.flags._replace(SYMMETRIC=True)
            if op.flags.ORTHOGONAL:
                op.flags = op.flags._replace(UNITARY=True)
            if op.flags.UNITARY:
                op.flags = op.flags._replace(ORTHOGONAL=True)

        if op.flags.ORTHOGONAL:
            if op.flags.IDEMPOTENT:
                op.flags = op.flags._replace(SYMMETRIC=True)
            if op.flags.SYMMETRIC:
                op.flags = op.flags._replace(IDEMPOTENT=True)

        if op.flags.UNITARY:
            if op.flags.IDEMPOTENT:
                op.flags = op.flags._replace(HERMITIAN=True)
            if op.flags.HERMITIAN:
                op.flags = op.flags._replace(IDEMPOTENT=True)

        if op.flags.INVOLUTARY:
            if op.flags.SYMMETRIC:
                op.flags = op.flags._replace(ORTHOGONAL=True)
            if op.flags.ORTHOGONAL:
                op.flags = op.flags._replace(SYMMETRIC=True)
            if op.flags.HERMITIAN:
                op.flags = op.flags._replace(UNITARY=True)
            if op.flags.UNITARY:
                op.flags = op.flags._replace(HERMITIAN=True)

        if op.flags.IDEMPOTENT:
            if any([op.flags.ORTHOGONAL, op.flags.UNITARY,
                    op.flags.INVOLUTARY]):
                op.flags = op.flags._replace(ORTHOGONAL=True, UNITARY=True,
                                             INVOLUTARY=True)

    def _set_rules(self):
        """ Translate flags into rules. """
        self.rules = {'addition':[], 'composition':[]}
        if self.flags.IDEMPOTENT:
            self.add_rule('..', '.')
        if self.flags.ORTHOGONAL:
            self.add_rule('.T.', '1')
        if self.flags.UNITARY:
            self.add_rule('.H.', '1')
        if self.flags.INVOLUTARY:
            self.add_rule('..', '1')

    def _set_inout(self, shapein, shapeout, reshapein, reshapeout):
        """
        Set methods and attributes dealing with the input and output handling.
        """
        shapein = tointtuple(shapein)
        shapeout = tointtuple(shapeout)

        if self.__class__.reshapein != Operator.reshapein:
            reshapein = self.reshapein
            self.reshapein = lambda v: Operator.reshapein(self, v)
        if reshapein is not None:
            self._reshapein = reshapein

        if self.__class__.reshapeout != Operator.reshapeout:
            reshapeout = self.reshapeout
            self.reshapeout = lambda v: Operator.reshapeout(self, v)
        if reshapeout is not None:
            self._reshapeout = reshapeout

        if shapein is shapeout is None:
            shapeout = self.reshapein(None)
            if shapeout is None and self._reshapein is None:
                self.flags = self.flags._replace(SQUARE=True)
            shapein = self.reshapeout(None)
        elif shapeout is not None and self._reshapein is not None:
            shapeout_ = tointtuple(self._reshapein(shapein))
            if shapeout_ is not None and shapeout_ != shapeout:
                raise ValueError("The specified output shape '{0}' is incompati"
                        "ble with that given by reshapein '{1}'.".format(
                        _strshape(shapeout), _strshape(shapeout_)))
        elif shapein is not None and self._reshapeout is not None:
            shapein_ = tointtuple(self._reshapeout(shapeout))
            if shapein_ is not None and shapein_ != shapein:
                raise ValueError("The specified input shape '{0}' is incompati"
                        "ble with that given by reshapeout '{1}'.".format(
                        _strshape(shapein), _strshape(shapein_)))
        elif shapein and shapeout is None and self._reshapein is None:
            self.flags = self.flags._replace(SQUARE=True)

        if self._reshapeout is not None:
            shapein = shapein or tointtuple(self._reshapeout(shapeout))
        if self._reshapein is not None:
            shapeout = shapeout or tointtuple(self._reshapein(shapein))

        if shapein is not None and shapein == shapeout:
            self.flags = self.flags._replace(SQUARE=True)

        if self.flags.SQUARE:
            shapeout = shapein
            self.reshapeout = self.reshapein
            self._reshapeout = self._reshapein
            self.toshapeout = self.toshapein
        self.shapein = shapein
        self.shapeout = shapeout
                    
    def _set_name(self):
        """Set operator's __name__ attribute."""
        if self.__class__ != 'Operator':
            name = self.__class__.__name__
        elif self.direct and self.direct.__name__ not in ('<lambda>', 'direct'):
            name = self.direct.__name__
        else:
            name = 'Operator'
        self.__name__ = name

    def _validate_input(self, input, output):
        """Return the input as ndarray subclass and allocate the output
        if necessary."""
        input = np.array(input, copy=False, subok=True, ndmin=1)
        if type(input) is np.ndarray:
            input = input.view(ndarraywrap)

        shapeout = self.reshapein(input.shape)
        dtype = self._find_common_type([input.dtype, self.dtype])
        input = np.array(input, dtype=dtype, subok=True, copy=False)
        if output is not None:
            if output.dtype != dtype:
                raise ValueError("Invalid output dtype '{0}'. Expected dtype is"
                                 " '{1}'.".format(output.dtype, dtype))
            if output.nbytes != np.product(shapeout) * dtype.itemsize:
                raise ValueError('The output has invalid shape {0}. Expected sh'
                                 'ape is {1}.'.format(output.shape, shapeout))

        output = self._allocate(shapeout, dtype, output)[0]
        return input, output

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return self.matvec(other)
        return CompositionOperator([self, other])

    def __rmul__(self, other):
        if not isscalar(other):
            raise NotImplementedError("It is not possible to multiply '" + \
                str(type(other)) + "' with an Operator.")
        return CompositionOperator([other, self])

    def __imul__(self, other):
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
        return ScalarOperator(-1) * self

    def __str__(self):
        if self.shapein is not None:
            if self.flags.SQUARE and len(self.shapein) > 1:
                s = _strshape(self.shapein) + '²'
            else:
                s = _strshape(self.shapeout) + 'x' + _strshape(self.shapein)
            s += ' '
        else:
            s = ''
        s += self.__name__
        return s

    def __repr__(self):
        a = []
        init = getattr(self, '__init_original__', self.__init__)
        vars, junk, junk, defaults = inspect.getargspec(init)
        for ivar, var in enumerate(vars):
            if var in ('flags', 'self'):
                continue
            val = getattr(self, var, None)
            if isinstance(val, types.MethodType):
                continue
            nargs = len(vars) - (len(defaults) if defaults is not None else 0)
            if ivar >= nargs:
                if val is defaults[ivar - nargs]:
                    continue
            if isinstance(val, Operator):
                s = 'Operator()'
            elif var in ['shapein', 'shapeout']:
                s = _strshape(val)
            elif isinstance(val, np.ndarray) and val.ndim == 0:
                s = repr(val[()])
            elif var == 'dtype':
                s = str(val)
            else:
                s = repr(val)
            if var == 'shapeout' and self.shapeout == self.shapein:
                continue
            if ivar < nargs:
                a += [ s ]
            else:
                a += [var + '=' + s]
        return self.__name__ + '(' + ', '.join(a) + ')'

def asoperator(operator, shapein=None, shapeout=None):
    if isinstance(operator, Operator):
        if shapein and operator.shapein and shapein != operator.shapein:
            raise ValueError('The input shapein ' + str(shapein) + ' is incompa'
                'atible with that of the input ' + str(operator.shapein) + '.')
        if shapeout and operator.shapeout and shapeout != operator.shapeout:
            raise ValueError('The input shapeout ' + str(shapeout) + ' is incom'
                'patible with that of the input ' + str(operator.shapeout) +  \
                '.')
        if shapein and not operator.shapein or \
           shapeout and not operator.shapeout:
            operator = copy.copy(operator)
            operator.shapein = shapein
            operator.shapeout = shapeout
        return operator

    if hasattr(operator, 'matvec') and hasattr(operator, 'rmatvec') and \
       hasattr(operator, 'shape'):
        def direct(input, output):
            output[:] = operator.matvec(input)
        def transpose(input, output):
            output[:] = operator.rmatvec(input)
        return Operator(direct=direct,
                        transpose=transpose,
                        shapein=shapein or operator.shape[1],
                        shapeout=shapeout or operator.shape[0],
                        dtype=operator.dtype,
                        flags={'LINEAR':True})
    
    if isscalar(operator):
        return ScalarOperator(operator)

    return asoperator(scipy.sparse.linalg.aslinearoperator(operator))


def asoperator1d(operator):
    operator = asoperator(operator)
    r = ReshapeOperator(operator.shape[1], operator.shapein)
    s = ReshapeOperator(operator.shapeout, operator.shape[0])
    return s * operator * r


class CompositeOperator(Operator):
    """
    Abstract class for grouping operands.
    """

    def __new__(cls, operands, *args, **keywords):
        operands = cls._validate_operands(operands)
        operands = cls._apply_rules(operands)
        if len(operands) == 1:
            return operands[0]
        instance = super(CompositeOperator, cls).__new__(cls)
        instance.operands = operands
        return instance

    @property
    def dtype(self):
        return self._find_common_type([op.dtype for op in self.operands])

    @dtype.setter
    def dtype(self, dtype):
        pass

    @classmethod
    def _apply_rules(cls, ops):
        return ops
        
    @classmethod
    def _validate_operands(cls, operands):
        operands = [asoperator(op) for op in operands]
        result = []
        for op in operands:
            if isinstance(op, cls):
                result.extend(op.operands)
            else:
                result.append(op)
        return result

    def __str__(self):
        if isinstance(self, AdditionOperator):
            op = ' + '
        elif isinstance(self, PartitionOperator):
            op = ' ⊕ '
        else:
            op = ' * '
        operands = ['({0})'.format(o) if isinstance(o, (AdditionOperator,
                    PartitionOperator)) else \
                    str(o) for o in self.operands]
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


class AdditionOperator(CompositeOperator):
    """
    Class for operator addition

    If at least one of the input already is the result of an addition,
    a flattened list of operators is created by associativity, to simplify
    reduction.
    """
    def __init__(self, operands):
        flags = {
            'LINEAR':all([op.flags.LINEAR for op in self.operands]),
            'REAL':all([op.flags.REAL for op in self.operands]),
            'SQUARE':self.shapein is not None and \
                self.shapein == self.shapeout or \
                all([op.flags.SQUARE for op in self.operands])}
        CompositeOperator.__init__(self, flags=flags)
        self.work = [None, None]

    def associated_operators(self):
        return { 'T' : AdditionOperator([m.T for m in self.operands]),
                 'H' : AdditionOperator([m.H for m in self.operands]),
                 'C' : AdditionOperator([m.conjugate() for m in self.operands]),
               }

    def direct(self, input, output):
        operands = self.operands

        # 1 operand: this case should not happen
        assert len(operands) > 1

        w0, new = self._allocate_like(output, self.work[0])
        if new:
            self.work[0] = w0
        self._propagate(output, w0)

        # 2 operands: 1 temporary
        if len(operands) == 2:
            operands[0].direct(input, output)
            w0.__class__ = output.__class__
            operands[1].direct(input, w0)
            output.__class__ = w0.__class__
            output += w0
            return

        # more than 2 operands, input == output: 2 temporaries
        if self.same_data(input, output):
            w1, new = self._allocate_like(output, self.work[1])
            if new:
                self.work[1] = w1
            operands[0].direct(input, w0)
            output.__class__ = w0.__class__
            self._propagate(w0, w1)
            for op in operands[1:-1]:
                op.direct(input, w1)
                output.__class__ = w1.__class__
                w0 += w1
            operands[-1].direct(input, output)
            output += w0
            return
        
        # more than 2 operands, input != output: 1 temporary
        operands[0].direct(input, output)
        self._propagate(output, w0)
        for op in self.operands[1:]:
            op.direct(input, w0)
            output.__class__ = w0.__class__
            output += w0

    def reshapein(self, shapein):
        shapeout = None
        for op in self.operands:
            shapeout_ = op.reshapein(shapein)
            if shapeout_ is None:
                continue
            if shapeout is None:
                shapeout = shapeout_
                continue
            if shapeout != shapeout_:
                raise ValueError("Incompatible shape in operands: '{0}' and '{1"
                                 "}'.".format(shapeout, shapeout_))
        return shapeout

    def reshapeout(self, shapeout):
        shapein = None
        for op in self.operands:
            shapein_ = op.reshapeout(shapeout)
            if shapein_ is None:
                continue
            if shapein is None:
                shapein = shapein_
                continue
            if shapein != shapein_:
                raise ValueError("Incompatible shape in operands: '{0}' and '{1"
                                 "}'.".format(shapein, shapein_))
        return shapein

    @staticmethod
    def _apply_rules(ops):
        if len(ops) <= 1:
            return ops
        i = 0
        while i < len(ops):
            j = 0
            consumed = False
            while j < len(ops):
                if j != i:
                    for rule in ops[i].rules['addition']:
                        new_ops = rule(ops[j])
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
        i = [i for i,o in enumerate(ops) if isinstance(o, ScalarOperator)]
        if len(i) > 0:
            ops.insert(0, ops[i[0]])
            del ops[i[0]+1]
            if ops[0].data == 0 and len(ops) > 1:
                del ops[0]
        return ops
                

class CompositionOperator(CompositeOperator):
    """
    Class handling operator composition.

    If at least one of the input already is the result of a composition,
    a flattened list of operators is created by associativity, to simplify
    reduction.
    """

    def __init__(self, operands):
        flags = {
            'LINEAR':all([op.flags.LINEAR for op in self.operands]),
            'REAL':all([op.flags.REAL for op in self.operands]),
            'SQUARE':self.shapein is not None and \
                (self.shapein == self.shapeout) or \
                all([op.flags.SQUARE for op in self.operands])}
        CompositeOperator.__init__(self, flags=flags)
        self.work = [None, None]

    def associated_operators(self):
        return {
            'C' : CompositionOperator([m.C for m in self.operands]),
            'T' : CompositionOperator([m.T for m in reversed(self.operands)]),
            'H' : CompositionOperator([m.H for m in reversed(self.operands)]),
            'I' : CompositionOperator([m.I for m in reversed(self.operands)]),
            'IC': CompositionOperator([m.I.C for m in reversed(self.operands)]),
            'IT': CompositionOperator([m.I.T for m in self.operands]),
            'IH': CompositionOperator([m.I.H for m in self.operands]),
        }

    def direct(self, input, output):

        operands = self.operands

        # 1 operand: this case should not happen
        assert len(operands) > 1

        # make the output buffer available in the work pool
        self._set_output(output)

        i = input
        for op in reversed(self.operands):
            # get output from the work pool
            o = self._get_output(op.reshapein(i.shape), output.dtype)
            op._propagate(output, o)
            op.direct(i, o)
            output.__class__ = o.__class__
            i = o

        # remove output from the work pool, to avoid side effects on the output
        self._del_output()

    def reshapein(self, shape):
        for op in reversed(self.operands):
            shape = op.reshapein(shape)
        return shape

    def reshapeout(self, shape):
        for op in self.operands:
            shape = op.reshapeout(shape)
        return shape

    def _get_output(self, shape, dtype):
        nbytes = np.product(shape) * dtype.itemsize
        if nbytes <= self.work[0].nbytes:
            return self.work[0][:nbytes].view(dtype).reshape(shape)

        buf, new = self._allocate(shape, dtype, self.work[1])
        if new:
            self.work[1] = buf
        return buf

    def _set_output(self, output):
        self.work[0] = output.ravel().view(np.int8, ndarraywrap)
        
    def _del_output(self):
        self.work[0] = None

    @staticmethod
    def _apply_rules(ops):
        if len(ops) <= 1:
            return ops
        i = len(ops) - 1

        while i >= 0:
            
            # inspect operators on the right
            consumed = False
            if i < len(ops) - 1:
                for rule in ops[i].rules['composition']:
                    if rule.reflected:
                        continue
                    new_ops = rule(ops[i+1])
                    if new_ops is None:
                        continue
                    consumed = True
                    if not isinstance(new_ops, tuple):
                        del ops[i+1]
                        ops[i] = new_ops
                    else:
                        raise NotImplementedError()
                    break

            if consumed:
                continue

            # inspect operators on the left
            if i > 0:
                for rule in ops[i].rules['composition']:
                    if not rule.reflected:
                        continue
                    new_ops = rule(ops[i-1])
                    if new_ops is None:
                        continue
                    consumed = True
                    if not isinstance(new_ops, tuple):
                        ops[i] = new_ops
                        del ops[i-1]
                        i -= 1
                    elif len(new_ops) == 2:
                        ops[i-1], ops[i] = new_ops
                    elif len(new_ops) == 3:
                        ops[i-1] = new_ops[0]
                        ops.insert(i, new_ops[1])
                        ops[i+1] = new_ops[2]
                        i += 1
                    else:
                        raise NotImplementedError()
                    break

            if consumed:
                continue

            i -= 1

        return ops


class PartitionOperator(CompositeOperator):
    """
    Block diagonal operator with more stringent conditions.

    The input and output shape of the block operators  must be the same, except
    for one same dimension: the axis along which the input is partitioned. This
    operator can be used to process data chunk by chunk.

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
    p = PartitionOperator([o1, o2], axis=-1)
    print p.shapein
    (16,7)

    """
    def __init__(self, operands, partitionin=None, axisin=0, axisout=None,
                 shapein=None, shapeout=None):
   
        if axisout is None:
            axisout = axisin

        if partitionin is None:
            partitionin = tuple(None if op.shapein is None else \
                op.shapein[axisin] for op in operands)
            if None in partitionin:
                partitionin = None
        partitionin = tointtuple(partitionin)

        if partitionin is not None:
            if len(partitionin) != len(operands):
                raise ValueError('The number of operators must be the same as t'
                                 'he length of the partition.')

        partitionout = self._get_partitionout(partitionin, axisin, axisout)

        if axisin >= 0:
            slicein = (axisin+1) * [slice(None)] + [Ellipsis]
        else:
            slicein = [Ellipsis] + (-axisin) * [slice(None)]
        if axisout >= 0:
            sliceout = (axisout+1) * [slice(None)] + [Ellipsis]
        else:
            sliceout = [Ellipsis] + (-axisout) * [slice(None)]

        flags = {
            'LINEAR':all([op.flags.LINEAR for op in self.operands]),
            'REAL':all([op.flags.REAL for op in self.operands]),
            'SQUARE':all([op.flags.SQUARE for op in self.operands])}

        self.axisin = axisin
        self.axisout = axisout
        self.partitionin = partitionin
        self.partitionout = partitionout
        self.slicein = slicein
        self.sliceout = sliceout
        CompositeOperator.__init__(self, shapein=shapein, shapeout=shapeout,
                                   flags=flags)
        self.add_rule('.{Operator}', self._rule_operator_comp_right)
        self.add_rule('{Operator}.', self._rule_operator_comp_left)
        self.add_rule('.{Operator}', self._rule_operator_add, 'addition')
        self.add_rule('{PartitionOperator}.',
            lambda o: self._rule_partition(o, self, np.add), 'addition')
        self.add_rule('{PartitionOperator}.',
            lambda o: self._rule_partition(o, self, np.multiply))

    def associated_operators(self):
        return {
            'C': PartitionOperator([op.C for op in self.operands],
                     self.partitionin, self.axisin, self.axisout),
            'T': PartitionOperator([op.T for op in self.operands],
                     self.partitionout, self.axisout, self.axisin),
            'H': PartitionOperator([op.H for op in self.operands],
                     self.partitionout, self.axisout, self.axisin),
            'I': PartitionOperator([op.I for op in self.operands],
                     self.partitionout, self.axisout, self.axisin),
            }
        
    def direct(self, input, output):
        if self.partitionout is None:
            shapeins = self._get_shapeins(input.shape)
            partitionout = [op.reshapein(s)[self.axisout] \
                            for op,s in zip(self.operands, shapeins)]
        else:
            partitionout = self.partitionout
        destin = 0
        destout = 0
        for op, nin, nout in zip(self.operands, self.partitionin, partitionout):
            self.slicein[self.axisin] = slice(destin, destin + nin)
            self.sliceout[self.axisout] = slice(destout, destout + nout)
            op.direct(input[self.slicein], output[self.sliceout])
            destin += nin
            destout += nout

    def reshapein(self, shapein):
        if shapein is None:
            shapeouts = [op.reshapein(None) for op in self.operands]
            return self._validate_shapes(shapeouts, self.partitionout,
                                         self.axisout)
        shapeouts = [op.reshapein(s) for op,s in zip(self.operands,
                     self._get_shapeins(shapein))]
        shapeout = list(shapeouts[0])
        shapeout[self.axisout] = np.sum((s[self.axisout] for s in shapeouts))
        return tointtuple(shapeout)

    def reshapeout(self, shapeout):
        if shapeout is None:
            shapeins = [op.reshapeout(None) for op in self.operands]
            return self._validate_shapes(shapeins, self.partitionin,
                                         self.axisin)
        if self.partitionout is None:
            raise ValueError('The input shape of an operator with implicit part'
                             'ition cannot be inferred.')
        shapeout0 = list(shapeout)
        shapeout0[self.axisout] = self.partitionout[0]
        shapein = list(self.operands[0].reshapeout(shapeout0))
        shapein[self.axisin] = np.sum(self.partitionin)
        return tointtuple(shapein)

    def toshapein(self, v):
        if self.shapein is not None:
            return v.reshape(self.shapein)
        if self.partitionin is None or self.axisin not in (0,-1):
            raise ValueError('Ambiguous reshaping.')
        p = np.sum(self.partitionin)
        if v.size == p:
            return v
        if self.axis == 0:
            return v.reshape((p,-1))
        return v.reshape((-1,p))

    def _get_shapeins(self, shapein):
        if self.partitionin is None:
            raise ValueError('The output shape of an operator with implicit par'
                             'tition cannot be inferred.')
        shapeins = []
        for p in self.partitionin:
            shapein_ = list(shapein)
            shapein_[self.axisin] = p
            shapeins.append(shapein_)
        return shapeins

    def _get_partitionout(self, partitionin, axisin, axisout):
        if partitionin is None:
            return None
        ndim_min = (axisin+1 if axisin >= 0 else -axisin)
        partitionout = len(self.operands) * [None]
        for i, op in enumerate(self.operands):
            pout = []
            # check that partitionout does not depend on the rank of the input
            for ndim in range(ndim_min, 33):
                shapein_ = ndim * [0]
                shapein_[axisin] = partitionin[i]
                try:
                    shapeout_ = op.shapeout or op.reshapein(shapein_)
                    pout.append(shapeout_[axisout])
                except (IndexError, NotImplementedError):
                    continue
            if len(pout) == 0 or any([p != pout[0] for p in pout]):
                return None
            partitionout[i] = pout[0]
        return tointtuple(partitionout)

    def _validate_shapes(self, shapes, p, axis):
        if p is None:
            return None
        explicit = [s is not None for s in shapes]
        try:
            s0 = shapes[explicit.index(True)]
        except ValueError:
            return None
        rank = len(s0)
        if any([s is not None and len(s) != rank for s in shapes]):
            raise ValueError('The partition operators do not have the same numb'
                             'er of dimensions.')
        if any([shapes[i] is not None and shapes[i][axis] != p[i] \
                for i in range(len(p))]):
            raise ValueError("The partition operators have shapes '{0}' incompa"
                "tible with the partition {1}.".format(
                _strshape(shapes), _strshape(p)))
        if np.sum(explicit) < 2:
            return None
        ok = [all([s is None or s[i] == s0[i] for s in shapes]) \
              for i in range(rank)]
        ok[axis] = True
        if not all(ok):
            raise ValueError("The dimensions of the partition operators '{0]' a"
                "re not the same along axes other than that of the partition." \
                .format(','.join([_strshape(s) for s in shapes])))
        if None in shapes or None in p:
            return None
        shape = list(s0)
        shape[axis] = np.sum(p)
        return tointtuple(shape)

    def _rule_operator_add(self, op):
        if op.shapein is not None:
            return None
        return PartitionOperator([o + op for o in self.operands],
            partitionin=self.partitionin, axisin=self.axisin)

    def _rule_operator_comp_left(self, op):
        if op.shapein is not None:
            return None
        return PartitionOperator([op * o for o in self.operands],
            partitionin=self.partitionin, axisin=self.axisin)

    def _rule_operator_comp_right(self, op):
        if op.shapein is not None:
            return None
        return PartitionOperator([o * op for o in self.operands],
            partitionin=self.partitionin, axisin=self.axisin)

    def _rule_partition(self, p1, p2, opn):
        if p1.axisin != p2.axisout:
            return None
        if p1.partitionin and p2.partitionout and \
           p1.partitionin != p2.partitionout:
            return None
        partition = p2.partitionin or p2.partitionin
        return PartitionOperator([opn(o1, o2) for o1,o2 in zip(p1.operands,
            p2.operands)], partitionin=partition, axisin=p2.axisin,
            axisout=p1.axisout)


class ConcatenationOperator(CompositeOperator):
    """
    Block column operator with more stringent conditions.

    Example
    -------
    >>> I = IdentityOperator(shapein=3)
    >>> op = ConcatenationOperator([I,2*I])
    >>> op.todense()

    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 2.,  0.,  0.],
           [ 0.,  2.,  0.],
           [ 0.,  0.,  2.]])

    """   
    def __init__(self, operands, partition=None, axis=0, shapein=None,
                 shapeout=None):
        if partition is None:
            partition = tuple(None if op.shapeout is None else \
                op.shapeout[axis] for op in operands)
            if None in partition:
                partition = None
        partition = tointtuple(partition)

        if partition is not None:
            if len(partition) != len(operands):
                raise ValueError('The number of operators must be the same as t'
                                 'he length of the partition.')

        if axis >= 0:
            slice_ = (axis+1) * [slice(None)] + [Ellipsis]
        else:
            slice_ = [Ellipsis] + (-axis) * [slice(None)]

        flags = {
            'LINEAR':all([op.flags.LINEAR for op in self.operands]),
            'REAL':all([op.flags.REAL for op in self.operands])}

        self.axis = axis
        self.partition = partition
        self.slice = slice_
        CompositeOperator.__init__(self, shapein=shapein, shapeout=shapeout,
                                   flags=flags)

    def direct(self, input, output):
        if self.partition is None:
            raise NotImplementedError()
        dest = 0
        for op, n in zip(self.operands, self.partition):
            self.slice[self.axis] = slice(dest, dest + n)
            op.direct(input, output[self.slice])
            dest += n

    def transpose(self, input, output):
        if self.partition is None:
            raise NotImplementedError()
        work = np.zeros_like(output)
        dest = 0
        for op, n in zip(self.operands, self.partition):
            self.slice[self.axis] = slice(dest, dest + n)
            op.T(input[self.slice], output)
            work += output
            dest += n
        output[:] = work

    def reshapein(self, shapein):
        shapeouts = [op.reshapein(shapein) for op in self.operands]
        shapeout = list(shapeouts[0])
        shapeout[self.axis] = np.sum((s[self.axis] for s in shapeouts))
        return shapeout

    def reshapeout(self, shapeout):
        return self.operands[0].shapein

    def __str__(self):
        operands = ['[{}]'.format(o) for o in self.operands]
        return '[' + ' '.join(operands) + ']'


@real
@orthogonal
class ReshapeOperator(Operator):
    """
    Operator that reshapes arrays.

    Example
    -------
    >>> op = ReshapeOperator(6, (3,2))
    >>> op(np.ones(6)).shape
    (3, 2)
    """
    def __new__(cls, shapein, shapeout):
        if shapein is None:
            raise ValueError('The input shape is None.')
        if shapeout is None:
            raise ValueError('The output shape is None.')
        shapein = tointtuple(shapein)
        shapeout = tointtuple(shapeout)
        if shapein == shapeout:
            return IdentityOperator(shapein)
        inst = super(ReshapeOperator, cls).__new__(cls)
        return inst

    def __init__(self, shapein, shapeout):
        if np.product(shapein) != np.product(shapeout):
            raise ValueError('The total size of the output must be unchanged.')
        Operator.__init__(self, shapein=shapein, shapeout=shapeout)

    def direct(self, input, output):
        if self.same_data(input, output):
            pass
        output.ravel()[:] = input.ravel()

    def associated_operators(self):
        return {'T':ReshapeOperator(self.shapeout, self.shapein)}

    def __str__(self):
        return _strshape(self.shapeout) + '←' + _strshape(self.shapein)


@symmetric
class ScalarOperator(Operator):
    """
    Multiplication by a scalar.

    """
    def __init__(self, data, shapein=None, dtype=None):
        if data is None:
            raise ValueError('Scalar value is None.')
        if not hasattr(data, '__add__') or not hasattr(data, '__mul__') or \
           not hasattr(data, '__cmp__') and not hasattr(data, '__eq__'):
            raise ValueError("Invalid scalar value '{0}'.".format(data))
        data = np.asarray(data)
        if dtype is None:
            dtype = np.find_common_type([data.dtype, float], [])
            data = np.array(data, dtype=dtype)

        if data == 0:
            flags = {'IDEMPOTENT':True}
        elif data in (1, -1):
            flags = {'IDEMPOTENT':True, 'INVOLUTARY':True}
        else:
            flags = None

        Operator.__init__(self, lambda i,o: np.multiply(i, data, o),
                          shapein=shapein, dtype=dtype, flags=flags)
        self.data = data
        self.add_rule('{Operator}.', self._rule_linear)
        self.add_rule('{ScalarOperator}.', self._rule_mul)
        self.add_rule('{ScalarOperator}.', self._rule_add, 'addition')

    def associated_operators(self):
        return {
            'C' : ScalarOperator(np.conjugate(self.data), shapein=self.shapein,
                                 dtype=self.dtype),
            'I' : ScalarOperator(1/self.data if self.data != 0 else np.nan,
                                 shapein=self.shapein, dtype=self.dtype),
            'IC' : ScalarOperator(np.conjugate(1/self.data) if self.data != 0 \
                                  else np.nan, shapein=self.shapein,
                                  dtype=self.dtype)
        }

    def __str__(self):
        data = self.data.flat[0]
        if data == int(data):
            data = int(data)
        return str(data)

    def _rule_linear(self, operator):
        if not operator.flags.LINEAR:
            return None
        if self.shapein is None or operator.shapein is not None:
            return (self, operator)
        return (ScalarOperator(self.data, dtype=self.dtype), operator,
                IdentityOperator(self.shapein))

    def _rule_add(self, s):
        return ScalarOperator(self.data + s.data, shapein=self.shapein or \
            s.shapein, dtype=self._find_common_type([self.dtype, s.dtype]))

    def _rule_mul(self, s):
        return ScalarOperator(self.data * s.data, shapein=self.shapein or \
            s.shapein, dtype=self._find_common_type([self.dtype, s.dtype]))


@real
@idempotent
@involutary
class IdentityOperator(ScalarOperator):
    """
    A subclass of ScalarOperator with value=1.
    All __init__ keyword arguments are passed to the
    ScalarOperator __init__.

    Exemple
    -------
    >>> I = IdentityOperator()
    >>> I.todense(3)

    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    >>> I = IdentityOperator(shapein=2)
    >>> I * arange(2)
    Info: Allocating (2,) float64 = 16 bytes in IdentityOperator.
    ndarraywrap([ 0.,  1.])

    """
    def __init__(self, shapein=None, dtype=None):
        ScalarOperator.__init__(self, 1, shapein=shapein, dtype=dtype)
        self.add_rule('.{Operator}', self._rule_identity)

    def direct(self, input, output):
        if self.same_data(input, output):
            pass
        output[:] = input

    def _rule_linear(self, operator):
        if not operator.flags.LINEAR:
            return None
        if self.shapein is None or operator.shapein is not None:
            return operator

    def _rule_identity(self, operator):
        if self.shapein is None or operator.shapeout is not None:
            return operator
        
@square
class BroadcastingOperator(Operator):
    """
    Abstract class for operators that operate on a data array and
    the input array, and for which broadcasting of the data array across
    the input array is required.
    """
    def __init__(self, data, broadcast='disabled', shapein=None, dtype=None,
                 **keywords):
        if data is None:
            raise ValueError('The data array is None.')

        if dtype is None:
            data = np.asarray(data)
            dtype = data.dtype
        self.data = np.array(data, dtype, copy=False, order='c', ndmin=1)

        broadcast = broadcast.lower()
        values = ('fast', 'slow', 'disabled')
        if broadcast not in values:
            raise ValueError("Invalid value '{0}' for the broadcast keyword. Ex"
                "pected values are {1}.".format(broadcast, strenum(values)))
        if broadcast == 'disabled':
            if shapein not in (None, self.data.shape):
                raise ValueError("The input shapein is incompatible with the da"
                                 "ta shape.")
            shapein = self.data.shape
        self.broadcast = broadcast

        Operator.__init__(self, shapein=shapein, dtype=dtype, **keywords)

    def reshapein(self, shape):
        if shape is None:
            return None
        n = self.data.ndim
        if len(shape) < n:
            raise ValueError("Invalid number of dimensions.")
        
        if self.broadcast == 'fast':
            it = zip(shape[:n], self.data.shape[:n])
        else:
            it = zip(shape[-n:], self.data.shape[-n:])
        for si, sd in it:
            if sd != 1 and sd != si:
                raise ValueError("The data array cannot be broadcast across the"
                                 " input.")
        return shape

    def toshapein(self, v):
        if self.shapein is not None:
            return v.reshape(self.shapein)
        if self.data.ndim < 2:
            return v

        sd = list(self.data.shape)
        n = sd.count(1)
        if n > 1:
            raise ValueError('Ambiguous broadcasting.')
        if n == 0:
            if self.broadcast == 'fast':
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


class ndarraywrap(np.ndarray):
    pass


def _strshape(shape):
    if len(shape) == 1:
        return str(shape[0])
    return str(shape).replace(' ','')
