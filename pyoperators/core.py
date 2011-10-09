# coding: utf-8
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

from collections import namedtuple
from . import memory
from .utils import isscalar, ndarraywrap, tointtuple, strenum, strshape
from .decorators import (
    real,
    idempotent,
    involutary,
    orthogonal,
    square,
    symmetric,
    inplace,
)

__all__ = [
    'Operator',
    'OperatorFlags',
    'AdditionOperator',
    'BroadcastingOperator',
    'CompositionOperator',
    'ExpansionOperator',
    'IdentityOperator',
    'PartitionOperator',
    'ReductionOperator',
    'ReshapeOperator',
    'ScalarOperator',
    'asoperator',
]


class OperatorFlags(
    namedtuple(
        'OperatorFlags',
        [
            'LINEAR',
            'SQUARE',  # shapein == shapeout
            'REAL',  # o.C = o
            'SYMMETRIC',  # o.T = o
            'HERMITIAN',  # o.H = o
            'IDEMPOTENT',  # o * o = o
            'ORTHOGONAL',  # o * o.T = I
            'UNITARY',  # o * o.H = I
            'INVOLUTARY',  # o * o = I
        ],
    )
):
    """Informative flags about the operator."""

    def __str__(self):
        n = max([len(f) for f in self._fields])
        fields = ['  ' + f.ljust(n) + ' : ' for f in self._fields]
        return '\n'.join([f + str(v) for f, v in zip(fields, self)])

    def __repr__(self):
        n = max([len(f) for f in self._fields])
        fields = [f.ljust(n) + '= ' for f in self._fields]
        return (
            self.__class__.__name__
            + '(\n  '
            + ',\n  '.join([f + str(v) for f, v in zip(fields, self)])
            + ')'
        )


class OperatorRule(object):
    """Binary rule on operators.

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
            '{self}': an instance of the reference operator's class
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
        if (
            isinstance(predicate, str)
            and predicate[0] == '{'
            and self.predicate[-1] == '}'
        ):
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
            if other_ == 'self':
                if not isinstance(other, self.reference.__class__):
                    return None
            elif other.__class__.__name__ != other_ and all(
                b.__name__ != other_ for b in other.__class__.__bases__
            ):
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
            return {
                '.': self.reference,
                '.C': self.reference.C,
                '.T': self.reference.T,
                '.H': self.reference.H,
                '.I': self.reference.I,
            }[symbol]
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

    def __init__(
        self,
        direct=None,
        transpose=None,
        adjoint=None,
        conjugate_=None,
        inverse=None,
        inverse_transpose=None,
        inverse_adjoint=None,
        inverse_conjugate=None,
        shapein=None,
        shapeout=None,
        reshapein=None,
        reshapeout=None,
        dtype=None,
        flags=None,
    ):

        for method, name in zip(
            (
                direct,
                transpose,
                adjoint,
                conjugate_,
                inverse,
                inverse_transpose,
                inverse_adjoint,
                inverse_conjugate,
            ),
            (
                'direct',
                'transpose',
                'adjoint',
                'conjugate_',
                'inverse',
                'inverse_transpose',
                'inverse_adjoint',
                'inverse_conjugate',
            ),
        ):
            if method is not None:
                if not hasattr(method, '__call__'):
                    raise TypeError("The method '%s' is not callable." % name)
                # should also check that the method has at least two arguments
                setattr(self, name, method)

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

        self._C = self._T = self._H = self._I = None

        self._set_dtype(dtype)
        self._set_flags(self, flags)
        self._set_rules()
        self._set_name()
        self._set_inout(shapein, shapeout, reshapein, reshapeout)

        if isinstance(self.direct, (types.FunctionType, types.MethodType)):
            if isinstance(self.direct, types.MethodType):
                d = self.direct.im_func
            else:
                d = self.direct
            self.inplace_reduction = 'operation' in d.func_code.co_varnames

    shapein = None
    shapeout = None
    dtype = None
    flags = OperatorFlags(*9 * (False,))
    inplace = False
    inplace_reduction = False
    _reshapein = None
    _reshapeout = None

    direct = None
    transpose = None
    adjoint = None

    def conjugate_(self, input, output):
        self.direct(input.conjugate(), output)
        output[...] = output.conjugate()

    inverse = None
    inverse_transpose = None
    inverse_adjoint = None

    def inverse_conjugate(self, input, output):
        self.inverse(input.conjugate(), output)
        output[...] = output.conjugate()

    def __call__(self, input, output=None):
        if self.direct is None:
            raise NotImplementedError(
                'Call to ' + self.__name__ + ' is not imp' 'lemented.'
            )
        i, o = self._validate_input(input, output)
        with memory.manager(o):
            if not self.inplace and self.same_data(i, o):
                memory.up()
                o_ = (
                    memory.get(o.nbytes, o.shape, o.dtype, self.__name__)
                    .view(o.dtype)
                    .reshape(o.shape)
                )
            else:
                o_ = o
            # o_.__dict__ = i.__dict__.copy()
            self.direct(i, o_)
            if not self.inplace and self.same_data(i, o):
                memory.down()
                o[...] = o_
                # o.__dict__ = o_.__dict__

        # o.__class__ = i.__class__ if type(o_) is ndarraywrap else o_.__class__
        if type(o) is ndarraywrap:
            if len(o.__dict__) == 0:
                o = o.base
        # elif o.__array_finalize__ is not None:
        #     d = o.__dict__.copy()
        #     o.__array_finalize__(None)
        #     for k, v in d.iteritems():
        #         setattr(o, k, v)
        return o

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
            raise ValueError(
                "The operator '" + self.__name__ + "' does not hav"
                "e an explicit shape."
            )
        return v.reshape(self.shapein)

    def toshapeout(self, v):
        """Reshape a vector into a multi-dimensional array compatible with
        the operator's output shape."""
        if self.shapeout is None:
            raise ValueError(
                "The operator '" + self.__name__ + "' does not hav"
                "e an explicit shape."
            )
        return v.reshape(self.shapeout)

    @staticmethod
    def same_data(array1, array2):
        return (
            array1.__array_interface__['data'][0]
            == array2.__array_interface__['data'][0]
        )

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
            raise ValueError(
                "The operator has an implicit shape. Use the 'shap" "ein' keyword."
            )
        shapeout = self.reshapein(shapein)
        m, n = np.product(shapeout), np.product(shapein)
        d = np.empty((n, m), self.dtype).view(ndarraywrap)
        v = np.zeros(n, self.dtype)
        for i in range(n):
            v[i] = 1
            o = d[i, :].reshape(shapeout)
            with memory.manager(o):
                self.direct(v.reshape(shapein), o)
            v[i] = 0
        if len(d.__dict__) == 0:
            d = d.view(np.ndarray)
        return d.T

    def matvec(self, v, output=None):
        v = self.toshapein(v)
        if output is not None:
            output = self.toshapeout(output)
        input, output = self._validate_input(v, output)
        with memory.manager(output):
            self.direct(input, output)
        return output.view(np.ndarray).ravel()

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

    def reshapein(self, shapein):
        """Return operator's output shape."""
        shapein = tointtuple(shapein)
        if None not in (self.shapein, shapein) and self.shapein != shapein:
            raise ValueError(
                "The input shape '{0}' is incompatible with that o"
                "f {1}: '{2}'.".format(
                    strshape(shapein), self.__name__, strshape(self.shapein)
                )
            )
        if self.shapeout is not None:
            return self.shapeout
        if self._reshapein is not None:
            return tointtuple(self._reshapein(shapein))
        if self.flags.SQUARE:
            return shapein
        return None

    def reshapeout(self, shapeout):
        """Return operator's input shape."""
        shapeout = tointtuple(shapeout)
        if None not in (self.shapeout, shapeout) and self.shapeout != shapeout:
            raise ValueError(
                "The output shape '{0}' is incompatible with that "
                "of {1}: '{2}'.".format(
                    strshape(shapeout), self.__name__, strshape(self.shapeout)
                )
            )
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
            raise ValueError(
                "Invalid associated operators. Expected operators "
                "are '{0}'".format(','.join(names))
            )

        if self.flags.REAL:
            C = self
        elif 'C' in ops:
            C = ops['C']
        else:
            C = Operator(
                self.conjugate_,
                shapein=self.shapein,
                shapeout=self.shapeout,
                reshapein=self._reshapein,
                reshapeout=self._reshapeout,
                dtype=self.dtype,
                flags=self.flags,
            )
            C.__name__ = self.__name__ + '.C'

        if self.flags.SYMMETRIC:
            T = self
        elif 'T' in ops:
            T = ops['T']
        else:
            T = Operator(
                self.transpose,
                shapein=self.shapeout,
                shapeout=self.shapein,
                reshapein=self._reshapeout,
                reshapeout=self._reshapein,
                dtype=self.dtype,
                flags=self.flags,
            )
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
            H = Operator(
                self.adjoint,
                shapein=self.shapeout,
                shapeout=self.shapein,
                reshapein=self._reshapeout,
                reshapeout=self._reshapein,
                dtype=self.dtype,
                flags=self.flags,
            )
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
            I = Operator(
                self.inverse,
                shapein=self.shapeout,
                shapeout=self.shapein,
                reshapein=self._reshapeout,
                reshapeout=self._reshapein,
                dtype=self.dtype,
                flags=self.flags,
            )
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
            IC = Operator(
                self.inverse_conjugate,
                shapein=self.shapeout,
                shapeout=self.shapein,
                reshapein=self._reshapeout,
                reshapeout=self._reshapein,
                dtype=self.dtype,
                flags=self.flags,
            )
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
            IT = Operator(
                self.inverse_transpose,
                shapein=self.shapein,
                shapeout=self.shapeout,
                reshapein=self._reshapein,
                reshapeout=self._reshapeout,
                dtype=self.dtype,
                flags=self.flags,
            )
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
            IH = Operator(
                self.inverse_adjoint,
                shapein=self.shapein,
                shapeout=self.shapeout,
                reshapein=self._reshapein,
                reshapeout=self._reshapeout,
                dtype=self.dtype,
                flags=self.flags,
            )
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
            if any([op.flags.ORTHOGONAL, op.flags.UNITARY, op.flags.INVOLUTARY]):
                op.flags = op.flags._replace(
                    ORTHOGONAL=True, UNITARY=True, INVOLUTARY=True
                )

    def _set_rules(self):
        """Translate flags into rules."""
        self.rules = {'addition': [], 'composition': []}
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
                raise ValueError(
                    "The specified output shape '{0}' is incompati"
                    "ble with that given by reshapein '{1}'.".format(
                        strshape(shapeout), strshape(shapeout_)
                    )
                )
        elif shapein is not None and self._reshapeout is not None:
            shapein_ = tointtuple(self._reshapeout(shapeout))
            if shapein_ is not None and shapein_ != shapein:
                raise ValueError(
                    "The specified input shape '{0}' is incompati"
                    "ble with that given by reshapeout '{1}'.".format(
                        strshape(shapein), strshape(shapein_)
                    )
                )
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
        input = np.array(input, copy=False, subok=True)
        if type(input) is np.ndarray:
            input = input.view(ndarraywrap)

        shapeout = self.reshapein(input.shape)
        dtype = self._find_common_type([input.dtype, self.dtype])
        input = np.array(input, dtype=dtype, subok=True, copy=False)
        if output is not None:
            if output.dtype != dtype:
                raise ValueError(
                    "The output has an invalid dtype '{0}'. Expect"
                    "ed dtype is '{1}'.".format(output.dtype, dtype)
                )
            if output.shape != shapeout:
                raise ValueError(
                    "The output has an invalid shape '{0}'. Expect"
                    "ed shape is '{1}'.".format(output.shape, shapeout)
                )
            output = output.view(ndarraywrap)
        else:
            output = memory.allocate(shapeout, dtype, None, self.__name__)[0]
        return input, output

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return self.matvec(other)
        return CompositionOperator([self, other])

    def __rmul__(self, other):
        if not isscalar(other):
            raise NotImplementedError(
                "It is not possible to multiply '"
                + str(type(other))
                + "' with an Operator."
            )
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
                s = strshape(self.shapein) + '²'
            else:
                s = strshape(self.shapeout) + 'x' + strshape(self.shapein)
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
                s = strshape(val)
            elif isinstance(val, np.ndarray) and val.ndim == 0:
                s = repr(val[()])
            elif var == 'dtype':
                s = str(val)
            else:
                s = repr(val)
            if var == 'shapeout' and self.shapeout == self.shapein:
                continue
            if ivar < nargs:
                a += [s]
            else:
                a += [var + '=' + s]
        return self.__name__ + '(' + ', '.join(a) + ')'


def asoperator(operator, shapein=None, shapeout=None):
    if isinstance(operator, Operator):
        if shapein and operator.shapein and shapein != operator.shapein:
            raise ValueError(
                'The input shapein ' + str(shapein) + ' is incompa'
                'atible with that of the input ' + str(operator.shapein) + '.'
            )
        if shapeout and operator.shapeout and shapeout != operator.shapeout:
            raise ValueError(
                'The input shapeout ' + str(shapeout) + ' is incom'
                'patible with that of the input ' + str(operator.shapeout) + '.'
            )
        if shapein and not operator.shapein or shapeout and not operator.shapeout:
            operator = copy.copy(operator)
            operator.shapein = shapein
            operator.shapeout = shapeout
        return operator

    if (
        hasattr(operator, 'matvec')
        and hasattr(operator, 'rmatvec')
        and hasattr(operator, 'shape')
    ):

        def direct(input, output):
            output[...] = operator.matvec(input)

        def transpose(input, output):
            output[...] = operator.rmatvec(input)

        return Operator(
            direct=direct,
            transpose=transpose,
            shapein=shapein or operator.shape[1],
            shapeout=shapeout or operator.shape[0],
            dtype=operator.dtype,
            flags={'LINEAR': True},
        )

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

    def __init__(self, operands, *args, **keywords):
        dtype = self._find_common_type([op.dtype for op in self.operands])
        Operator.__init__(self, dtype=dtype, **keywords)

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
        operands = [
            '({0})'.format(o)
            if isinstance(o, (AdditionOperator, PartitionOperator))
            else str(o)
            for o in self.operands
        ]
        if isinstance(self, PartitionOperator):
            if len(operands) > 2:
                operands = [operands[0], '...', operands[-1]]
        return op.join(operands)

    def __repr__(self):
        r = self.__name__ + '(['
        rops = [repr(op) for op in self.operands]
        components = []
        for i, rop in enumerate(rops):
            if i != len(rops) - 1:
                rop += ','
            components.extend(rop.split('\n'))
        r += '\n    ' + '\n    '.join(components) + '])'
        return r


class AdditionOperator(CompositeOperator):
    """
    Class for operator addition

    If at least one of the input already is the result of an addition,
    a flattened list of operators is created by associativity, to simplify
    reduction.
    """

    def __init__(self, operands):
        try:
            index = [o.inplace_reduction for o in operands].index(False)
            o = operands.pop(index)
            operands.insert(0, o)
        except ValueError:
            pass
        flags = {
            'LINEAR': all([op.flags.LINEAR for op in self.operands]),
            'REAL': all([op.flags.REAL for op in self.operands]),
            'SQUARE': self.shapein is not None
            and self.shapein == self.shapeout
            or all([op.flags.SQUARE for op in self.operands]),
        }
        CompositeOperator.__init__(self, operands, flags=flags)
        self.need_temporary = any(not o.inplace_reduction for o in operands[1:])

    def associated_operators(self):
        return {
            'T': AdditionOperator([m.T for m in self.operands]),
            'H': AdditionOperator([m.H for m in self.operands]),
            'C': AdditionOperator([m.conjugate() for m in self.operands]),
        }

    def direct(self, input, output):
        operands = self.operands
        assert len(operands) > 1

        if self.need_temporary:
            memory.up()
            buf = (
                memory.get(output.nbytes, output.shape, output.dtype, self.__name__)
                .view(output.dtype)
                .reshape(output.shape)
            )

        operands[0].direct(input, output)

        for op in operands[1:]:
            if op.inplace_reduction:
                op.direct(input, output, operation=operator.__iadd__)
            else:
                op.direct(input, buf)
                output += buf

        if self.need_temporary:
            memory.down()

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
                raise ValueError(
                    "Incompatible shape in operands: '{0}' and '{1"
                    "}'.".format(shapeout, shapeout_)
                )
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
                raise ValueError(
                    "Incompatible shape in operands: '{0}' and '{1"
                    "}'.".format(shapein, shapein_)
                )
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
        i = [i for i, o in enumerate(ops) if isinstance(o, ScalarOperator)]
        if len(i) > 0:
            ops.insert(0, ops[i[0]])
            del ops[i[0] + 1]
            if ops[0].data == 0 and len(ops) > 1:
                del ops[0]
        return ops


@inplace
class CompositionOperator(CompositeOperator):
    """
    Class handling operator composition.

    If at least one of the input already is the result of a composition,
    a flattened list of operators is created by associativity, to simplify
    reduction.
    """

    def __init__(self, operands):
        flags = {
            'LINEAR': all([op.flags.LINEAR for op in self.operands]),
            'REAL': all([op.flags.REAL for op in self.operands]),
            'SQUARE': self.shapein is not None
            and (self.shapein == self.shapeout)
            or all([op.flags.SQUARE for op in self.operands]),
        }
        CompositeOperator.__init__(self, operands, flags=flags)
        self.inplace_reduction = self.operands[0].inplace_reduction
        self._info = {}

    def associated_operators(self):
        return {
            'C': CompositionOperator([m.C for m in self.operands]),
            'T': CompositionOperator([m.T for m in reversed(self.operands)]),
            'H': CompositionOperator([m.H for m in reversed(self.operands)]),
            'I': CompositionOperator([m.I for m in reversed(self.operands)]),
            'IC': CompositionOperator([m.I.C for m in reversed(self.operands)]),
            'IT': CompositionOperator([m.I.T for m in self.operands]),
            'IH': CompositionOperator([m.I.H for m in self.operands]),
        }

    def direct(self, input, output, operation=None):

        inplace_composition = self.same_data(input, output)
        shapeouts, sizeouts, outplaces, reuse_output = self._get_info(
            input.shape,
            output.nbytes,
            output.dtype,
            inplace_composition and operation is None,
        )
        noutplaces = outplaces.count(True)
        # new_class = None

        nswaps = 0
        if not reuse_output:
            memory.up()
        elif (
            inplace_composition
            and outplaces[-1]
            or not inplace_composition
            and noutplaces % 2 == 0
        ):
            memory.swap()
            nswaps += 1

        # def do_direct(op, i, sizeout, shapeout, dtype):
        #    o = memory.get(sizeout, shapeout, dtype, self.__name__).view(dtype).reshape(shapeout)
        #    o.__dict__ = output.__dict__
        #    op.direct(i, o)
        #    if o.__class__ is not ndarraywrap:
        #        new_class = o.__class__
        #        print 'changing class', new_class
        #        o.__class__ = ndarraywrap
        #    return o, new_class

        i = input
        for iop, (op, shapeout, sizeout, outplace) in enumerate(
            zip(self.operands, shapeouts, sizeouts, outplaces)[:0:-1]
        ):
            if outplace and iop > 0:
                # input and output must be different
                memory.up()
                # i = do_direct(op, i, sizeout, shapeout, output.dtype)
                o = (
                    memory.get(sizeout, shapeout, output.dtype, self.__name__)
                    .view(output.dtype)
                    .reshape(shapeout)
                )
                op.direct(i, o)
                i = o
                memory.down()
                memory.swap()
                nswaps += 1
            else:
                # we keep reusing the same stack element for inplace operators
                # i = do_direct(op, i, sizeout, shapeout, output.dtype)
                o = (
                    memory.get(sizeout, shapeout, output.dtype, self.__name__)
                    .view(output.dtype)
                    .reshape(shapeout)
                )
                op.direct(i, o)
                i = o

        if outplaces[0]:
            memory.up()
        if self.inplace_reduction:
            self.operands[0].direct(i, output, operation=operation)
        else:
            self.operands[0].direct(i, output)
        # print 'new_class', new_class
        # print 'output.__class__', output.__class__
        # if new_class is not None and output.__class__ is ndarraywrap:
        #     output.__class__ = new_class
        if outplaces[0]:
            memory.down()
            memory.swap()
            nswaps += 1

        if nswaps % 2 == 1:
            memory.swap()

        if not reuse_output:
            memory.down()

    def reshapein(self, shape):
        for op in reversed(self.operands):
            shape = op.reshapein(shape)
        return shape

    def reshapeout(self, shape):
        for op in self.operands:
            shape = op.reshapeout(shape)
        return shape

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
                    new_ops = rule(ops[i + 1])
                    if new_ops is None:
                        continue
                    consumed = True
                    if not isinstance(new_ops, tuple):
                        del ops[i + 1]
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
                    new_ops = rule(ops[i - 1])
                    if new_ops is None:
                        continue
                    consumed = True
                    if not isinstance(new_ops, tuple):
                        ops[i] = new_ops
                        del ops[i - 1]
                        i -= 1
                    elif len(new_ops) == 2:
                        ops[i - 1], ops[i] = new_ops
                    elif len(new_ops) == 3:
                        ops[i - 1] = new_ops[0]
                        ops.insert(i, new_ops[1])
                        ops[i + 1] = new_ops[2]
                        i += 1
                    else:
                        raise NotImplementedError()
                    break

            if consumed:
                continue

            i -= 1

        return ops

    def _get_info(self, shape, nbytes, dtype, inplace):
        try:
            return self._info[(shape, nbytes, dtype, inplace)]
        except KeyError:
            pass
        shapeouts = self._get_shapeouts(shape)
        sizeouts = self._get_sizeouts(shapeouts)
        outplaces, reuse_output = self._get_outplaces(nbytes, inplace, sizeouts)
        v = shapeouts, sizeouts, outplaces, reuse_output
        self._info[(shape, nbytes, dtype, inplace)] = v
        return v

    def _get_shapeouts(self, shapein):
        if shapein is None:
            shapein = self.shapein
        shapeouts = []
        for op in reversed(self.operands):
            shapein = op.reshapein(shapein)
            if shapein is None:
                return None
            shapeouts.insert(0, shapein)
        return shapeouts

    def _get_sizeouts(self, shapeouts):
        # assuming input's dtype is float64
        if shapeouts is None:
            return None
        sizeouts = []
        dtype = np.dtype(np.float64)
        for op, shapeout in reversed(zip(self.operands, shapeouts)):
            dtype = self._find_common_type([dtype, op.dtype])
            sizeouts.insert(0, dtype.itemsize * np.prod(shapeout))
        return sizeouts

    def _get_outplaces(self, output_nbytes, inplace_composition, sizeouts):
        outplaces = [not op.inplace for op in self.operands]
        if not inplace_composition:
            outplaces[-1] = True

        noutplaces = outplaces.count(True)
        if (
            inplace_composition
            and noutplaces % 2 == 1
            and noutplaces == len(self.operands)
        ):
            return outplaces, False

        last_inplace_changed_to_outplace = False
        if inplace_composition:
            # if composition is inplace, enforce  even number of outplace
            if noutplaces % 2 == 1 and False in outplaces:
                index = outplaces.index(False)
                outplaces[index] = True
                last_inplace_changed_to_outplace = True
            output_is_requested = True  # we start with the input=output
        else:
            output_is_requested = noutplaces % 2 == 0

        reuse_output = False
        for op, outplace, nbytes in zip(self.operands, outplaces, sizeouts)[:0:-1]:
            if outplace:
                output_is_requested = not output_is_requested
            if output_is_requested:
                if nbytes > output_nbytes:
                    if last_inplace_changed_to_outplace:
                        outplaces[index] = False  # revert back
                    return outplaces, False
                reuse_output = True
        return outplaces, reuse_output


class PartitionBaseOperator(CompositeOperator):
    """
    Abstract base class for PartitionOperator, ExpansionOperator and
    ReductionOperator.
    """

    def __init__(
        self, operands, partitionin=None, partitionout=None, axisin=None, axisout=None
    ):
        if partitionin is partitionout is None:
            raise ValueError('No partition is provided.')
        if partitionin is not None:
            if len(partitionin) != len(operands):
                raise ValueError(
                    'The number of operators must be the same as t'
                    'he length of the input partition.'
                )
        if partitionout is not None:
            if len(partitionout) != len(operands):
                raise ValueError(
                    'The number of operators must be the same as t'
                    'he length of the output partition.'
                )
        flags = {
            'LINEAR': all([op.flags.LINEAR for op in self.operands]),
            'REAL': all([op.flags.REAL for op in self.operands]),
        }

        if partitionin is not None and partitionout is not None:
            flags['SQUARE'] = all([op.flags.SQUARE for op in self.operands])

        self.axisin = axisin
        self.axisout = axisout
        self.partitionin = partitionin
        self.partitionout = partitionout
        self.slicein = self._get_slice(axisin)
        self.sliceout = self._get_slice(axisout)
        if partitionin is None:
            self.__class__ = ExpansionOperator
        elif partitionout is None:
            self.__class__ = ReductionOperator
        else:
            self.__class__ = PartitionOperator
        CompositeOperator.__init__(self, operands, flags=flags)
        self.add_rule('.{Operator}', self._rule_operator_add, 'addition')
        self.add_rule('.{self}', self._rule_add, 'addition')
        self.add_rule('.{Operator}', self._rule_operator_comp_right)
        self.add_rule('{Operator}.', self._rule_operator_comp_left)
        self.add_rule('.{PartitionBaseOperator}', self._rule_comp_right)
        self.add_rule('{PartitionBaseOperator}.', self._rule_comp_left)

    def reshapein(self, shapein):
        if shapein is None:
            shapeouts = [op.reshapein(None) for op in self.operands]
        elif self.partitionin is None:
            shapeouts = [op.reshapein(shapein) for op in self.operands]
        else:
            shapeouts = [
                op.reshapein(s)
                for op, s in zip(
                    self.operands,
                    self._get_shapes(shapein, self.partitionin, self.axisin),
                )
            ]
        s = self._validate_shapes(shapeouts, self.partitionout, self.axisout)
        if None in shapeouts or s is None:
            if shapein is None:
                return None
            raise ValueError('Ambiguous implicit partition.')
        if self.partitionout is None:
            return s
        shapeout = list(s)
        shapeout[self.axisout] = np.sum([s[self.axisout] for s in shapeouts])
        return tointtuple(shapeout)

    def reshapeout(self, shapeout):
        if shapeout is None:
            shapeins = [op.reshapeout(None) for op in self.operands]
        elif self.partitionout is None:
            shapeins = [op.reshapeout(shapeout) for op in self.operands]
        else:
            shapeins = [
                op.reshapeout(s)
                for op, s in zip(
                    self.operands,
                    self._get_shapes(shapeout, self.partitionout, self.axisout),
                )
            ]
        s = self._validate_shapes(shapeins, self.partitionin, self.axisin)
        if None in shapeins or s is None:
            if shapeout is None:
                return None
            raise ValueError('Ambiguous implicit partition.')
        if self.partitionin is None:
            return s
        shapein = list(s)
        shapein[self.axisin] = np.sum([s[self.axisin] for s in shapeins])
        return tointtuple(shapein)

    def toshapein(self, v):
        if self.shapein is not None:
            return v.reshape(self.shapein)
        if self.partitionin is None:
            return self.operands[0].toshapein(v)
        if None in self.partitionin or self.axisin not in (0, -1):
            raise ValueError('Ambiguous reshaping.')
        p = np.sum(self.partitionin)
        if v.size == p:
            return v
        if self.axisin == 0:
            return v.reshape((p, -1))
        return v.reshape((-1, p))

    def toshapeout(self, v):
        if self.shapeout is not None:
            return v.reshape(self.shapeout)
        if self.partitionout is None:
            return self.operands[0].toshapeout(v)
        if None in self.partitionout or self.axisout not in (0, -1):
            raise ValueError('Ambiguous reshaping.')
        p = np.sum(self.partitionout)
        if v.size == p:
            return v
        if self.axisout == 0:
            return v.reshape((p, -1))
        return v.reshape((-1, p))

    @staticmethod
    def _get_partition(shapes, axis):
        return tuple(None if s is None else s[axis] for s in shapes)

    @staticmethod
    def _get_partitionin(ops, partitionout, axisout, axisin):
        ndim_min = axisout + 1 if axisout >= 0 else -axisout
        partitionin = len(ops) * [None]
        for i, op in enumerate(ops):
            if partitionout[i] is None:
                continue
            pin = []
            # check that partitionout does not depend on the rank of the input
            for ndim in range(ndim_min, 33):
                shapeout = ndim * [0]
                shapeout[axisout] = partitionout[i]
                try:
                    shapein = op.shapein or op.reshapeout(shapeout)
                    pin.append(shapein[axisin])
                except IndexError:
                    continue
            if len(pin) == 0 or any([p != pin[0] for p in pin]):
                continue
            partitionin[i] = pin[0]
        return tuple(partitionin)

    @staticmethod
    def _get_partitionout(ops, partitionin, axisin, axisout):
        ndim_min = axisin + 1 if axisin >= 0 else -axisin
        partitionout = len(ops) * [None]
        for i, op in enumerate(ops):
            if partitionin[i] is None:
                continue
            pout = []
            # check that partitionout does not depend on the rank of the input
            for ndim in range(ndim_min, 33):
                shapein = ndim * [0]
                shapein[axisin] = partitionin[i]
                try:
                    shapeout = op.shapeout or op.reshapein(shapein)
                    pout.append(shapeout[axisout])
                except IndexError:
                    continue
            if len(pout) == 0 or any([p != pout[0] for p in pout]):
                continue
            partitionout[i] = pout[0]
        return tuple(partitionout)

    @staticmethod
    def _get_shapes(shape, partition, axis):
        if None in partition:
            raise ValueError(
                'The shape of an operator with implicit partition '
                'cannot be inferred.'
            )
        shapes = []
        for p in partition:
            shape_ = list(shape)
            shape_[axis] = p
            shapes.append(shape_)
        return shapes

    @staticmethod
    def _get_slice(axis):
        if axis is None:
            return None
        if axis >= 0:
            return (axis + 1) * [slice(None)] + [Ellipsis]
        return [Ellipsis] + (-axis) * [slice(None)]

    @staticmethod
    def _validate_composition(op1, op2):
        if op1.axisin != op2.axisout:
            return None
        p1 = op1.partitionin
        p2 = op2.partitionout
        if p1 is None or p2 is None:
            return None
        if len(p1) != len(p2):
            return None
        if any(p != q for p, q in zip(p1, p2) if None not in (p, q)):
            return None
        return op2.partitionin, op1.partitionout

    @staticmethod
    def _validate_addition(op1, op2):
        if op1.axisin != op2.axisin or op1.axisout != op2.axisout:
            return None, None

        def func(p1, p2):
            if p1 is None and p2 is not None or p1 is not None and p2 is None:
                return None
            if len(p1) != len(p2):
                return None
            if any(p != q for p, q in zip(p1, p2) if None not in (p, q)):
                return None
            return [p or q for p, q in zip(p1, p2)]

        return func(op1.partitionin, op2.partitionin), func(
            op1.partitionout, op2.partitionout
        )

    @staticmethod
    def _validate_shapes(shapes, p, axis):
        if p is None:
            if any([s != shapes[0] for s in shapes]):
                raise ValueError('The operands have incompatible shapes.')
            return shapes[0]
        explicit = [s is not None for s in shapes]
        try:
            s0 = shapes[explicit.index(True)]
        except ValueError:
            return None
        rank = len(s0)
        if any([s is not None and len(s) != rank for s in shapes]):
            raise ValueError(
                'The partition operators do not have the same numb' 'er of dimensions.'
            )
        if any(
            [
                shapes[i] is not None and shapes[i][axis] != p[i]
                for i in range(len(p))
                if p[i] is not None
            ]
        ):
            raise ValueError(
                "The partition operators have shapes '{0}' incompa"
                "tible with the partition {1}.".format(strshape(shapes), strshape(p))
            )
        if np.sum(explicit) < 2:
            return s0
        ok = [all([s is None or s[i] == s0[i] for s in shapes]) for i in range(rank)]
        ok[axis] = True
        if not all(ok):
            raise ValueError(
                "The dimensions of the partition operators '{0]' a"
                "re not the same along axes other than that of the partition.".format(
                    ','.join([strshape(s) for s in shapes])
                )
            )
        return s0

    def _rule_operator_add(self, op):
        if op.shapein is not None:
            return None
        return PartitionBaseOperator(
            [o + op for o in self.operands],
            partitionin=self.partitionin,
            axisin=self.axisin,
            partitionout=self.partitionout,
            axisout=self.axisout,
        )

    def _rule_add(self, p):
        partitionin, partitionout = self._validate_addition(p, self)
        if partitionin is partitionout is None:
            return None
        operands = [o1 + o2 for o1, o2 in zip(p.operands, self.operands)]
        return PartitionBaseOperator(
            operands,
            partitionin=partitionin,
            axisin=self.axisin,
            partitionout=partitionout,
            axisout=self.axisout,
        )

    def _rule_operator_comp_left(self, op):
        if self.partitionout is None:
            return None
        if op.shapein is not None:
            return None
        n = len(self.partitionout)
        partitionout = self._get_partitionout(
            n * [op], self.partitionout, self.axisout, self.axisout
        )
        return PartitionBaseOperator(
            [op * o for o in self.operands],
            partitionin=self.partitionin,
            axisin=self.axisin,
            partitionout=partitionout,
            axisout=self.axisout,
        )

    def _rule_operator_comp_right(self, op):
        if self.partitionin is None:
            return None
        if op.shapein is not None:
            return None
        n = len(self.partitionin)
        partitionin = self._get_partitionin(
            n * [op], self.partitionin, self.axisin, self.axisin
        )
        return PartitionBaseOperator(
            [o * op for o in self.operands],
            partitionin=partitionin,
            axisin=self.axisin,
            partitionout=self.partitionout,
            axisout=self.axisout,
        )

    def _rule_comp_left(self, p):
        return self._rule_comp(p, self)

    def _rule_comp_right(self, p):
        return self._rule_comp(self, p)

    def _rule_comp(self, p1, p2):
        partitions = self._validate_composition(p1, p2)
        if partitions is None:
            return None
        partitionin, partitionout = partitions
        axisin, axisout = p2.axisin, p1.axisout
        operands = [o1 * o2 for o1, o2 in zip(p1.operands, p2.operands)]
        if partitionin is partitionout is None:
            return AdditionOperator(operands)
        return PartitionBaseOperator(
            operands,
            partitionin=partitionin,
            axisin=axisin,
            partitionout=partitionout,
            axisout=axisout,
        )


class PartitionOperator(PartitionBaseOperator):
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
    print(p.shapein)
    (16,7)

    """

    def __init__(self, operands, partitionin=None, axisin=0, axisout=None):

        if axisout is None:
            axisout = axisin

        if partitionin is None:
            partitionin = self._get_partition([op.shapein for op in operands], axisin)
        partitionin = tointtuple(partitionin)
        partitionout = self._get_partitionout(operands, partitionin, axisin, axisout)

        PartitionBaseOperator.__init__(
            self,
            operands,
            partitionin=partitionin,
            partitionout=partitionout,
            axisin=axisin,
            axisout=axisout,
        )

    def associated_operators(self):
        return {
            'C': PartitionOperator(
                [op.C for op in self.operands],
                self.partitionin,
                self.axisin,
                self.axisout,
            ),
            'T': PartitionOperator(
                [op.T for op in self.operands],
                self.partitionout,
                self.axisout,
                self.axisin,
            ),
            'H': PartitionOperator(
                [op.H for op in self.operands],
                self.partitionout,
                self.axisout,
                self.axisin,
            ),
            'I': PartitionOperator(
                [op.I for op in self.operands],
                self.partitionout,
                self.axisout,
                self.axisin,
            ),
        }

    def direct(self, input, output):
        if None in self.partitionout:
            shapeins = self._get_shapes(input.shape, self.partitionin, self.axisin)
            partitionout = [
                op.reshapein(s)[self.axisout] for op, s in zip(self.operands, shapeins)
            ]
        else:
            partitionout = self.partitionout
        destin = 0
        destout = 0
        for op, nin, nout in zip(self.operands, self.partitionin, partitionout):
            self.slicein[self.axisin] = slice(destin, destin + nin)
            self.sliceout[self.axisout] = slice(destout, destout + nout)
            input_ = input[self.slicein]
            output_ = output[self.sliceout]
            # output_.__dict__ = output.__dict__
            op.direct(input_, output_)
            # if output_.__class__ is not ndarraywrap:
            #    output.__class__ = output_.__class__
            destin += nin
            destout += nout


class ExpansionOperator(PartitionBaseOperator):
    """
    Block column operator with more stringent conditions.

    Example
    -------
    >>> I = IdentityOperator(shapein=3)
    >>> op = ExpansionOperator([I,2*I])
    >>> op.todense()

    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 2.,  0.,  0.],
           [ 0.,  2.,  0.],
           [ 0.,  0.,  2.]])

    """

    def __init__(self, operands, partitionout=None, axisout=0):
        if partitionout is None:
            partitionout = self._get_partition(
                [op.shapeout for op in operands], axisout
            )
        partitionout = tointtuple(partitionout)

        PartitionBaseOperator.__init__(
            self, operands, partitionout=partitionout, axisout=axisout
        )

    def associated_operators(self):
        return {
            'C': ExpansionOperator(
                [op.C for op in self.operands], self.partitionout, self.axisout
            ),
            'T': ReductionOperator(
                [op.T for op in self.operands], self.partitionout, self.axisout
            ),
            'H': ReductionOperator(
                [op.H for op in self.operands], self.partitionout, self.axisout
            ),
        }

    def direct(self, input, output):
        if None in self.partitionout:
            raise NotImplementedError()
        dest = 0
        for op, n in zip(self.operands, self.partitionout):
            self.sliceout[self.axisout] = slice(dest, dest + n)
            op.direct(input, output[self.sliceout])
            dest += n

    def __str__(self):
        operands = ['[{}]'.format(o) for o in self.operands]
        if len(operands) > 2:
            operands = [operands[0], '...', operands[-1]]
        return '[ ' + ' '.join(operands) + ' ]'


class ReductionOperator(PartitionBaseOperator):
    """
    Block row operator with more stringent conditions.

    Example
    -------
    >>> I = IdentityOperator(shapein=3)
    >>> op = ReductionOperator([I,2*I])
    >>> op.todense()

    array([[ 1.,  0.,  0., 2., 0., 0.],
           [ 0.,  1.,  0., 0., 2., 0.],
           [ 0.,  0.,  1., 0., 0., 2.]])

    """

    def __init__(self, operands, partitionin=None, axisin=0):
        if partitionin is None:
            partitionin = self._get_partition([op.shapein for op in operands], axisin)
        partitionin = tointtuple(partitionin)

        PartitionBaseOperator.__init__(
            self, operands, partitionin=partitionin, axisin=axisin
        )

    def associated_operators(self):
        return {
            'C': ReductionOperator(
                [op.C for op in self.operands], self.partitionin, self.axisin
            ),
            'T': ExpansionOperator(
                [op.T for op in self.operands], self.partitionin, self.axisin
            ),
            'H': ExpansionOperator(
                [op.H for op in self.operands], self.partitionin, self.axisin
            ),
        }

    def direct(self, input, output):
        if None in self.partitionin:
            raise NotImplementedError()
        work = np.zeros_like(output)
        dest = 0
        for op, n in zip(self.operands, self.partitionin):
            self.slicein[self.axisin] = slice(dest, dest + n)
            op(input[self.slicein], output)
            work += output
            dest += n
        output[...] = work

    def __str__(self):
        operands = [str(o) for o in self.operands]
        if len(operands) > 2:
            operands = [operands[0], '...', operands[-1]]
        return '[[ ' + ' '.join(operands) + ' ]]'


@real
@orthogonal
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
        return {'T': ReshapeOperator(self.shapeout, self.shapein)}

    def __str__(self):
        return strshape(self.shapeout) + '←' + strshape(self.shapein)


@symmetric
@inplace
class ScalarOperator(Operator):
    """
    Multiplication by a scalar.

    """

    def __init__(self, data, shapein=None, dtype=None):
        if data is None:
            raise ValueError('Scalar value is None.')
        if (
            not hasattr(data, '__add__')
            or not hasattr(data, '__mul__')
            or not hasattr(data, '__cmp__')
            and not hasattr(data, '__eq__')
        ):
            raise ValueError("Invalid scalar value '{0}'.".format(data))
        data = np.asarray(data)
        if dtype is None:
            dtype = np.find_common_type([data.dtype, float], [])
            data = np.array(data, dtype=dtype)

        if data == 0:
            flags = {'IDEMPOTENT': True}
        elif data in (1, -1):
            flags = {'IDEMPOTENT': True, 'INVOLUTARY': True}
        else:
            flags = None

        Operator.__init__(
            self,
            lambda i, o: np.multiply(i, data, o),
            shapein=shapein,
            dtype=dtype,
            flags=flags,
        )
        self.data = data
        self.add_rule('{Operator}.', self._rule_linear)
        self.add_rule('{ScalarOperator}.', self._rule_mul)
        self.add_rule('{ScalarOperator}.', self._rule_add, 'addition')

    def associated_operators(self):
        return {
            'C': ScalarOperator(
                np.conjugate(self.data), shapein=self.shapein, dtype=self.dtype
            ),
            'I': ScalarOperator(
                1 / self.data if self.data != 0 else np.nan,
                shapein=self.shapein,
                dtype=self.dtype,
            ),
            'IC': ScalarOperator(
                np.conjugate(1 / self.data) if self.data != 0 else np.nan,
                shapein=self.shapein,
                dtype=self.dtype,
            ),
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
        return (
            ScalarOperator(self.data, dtype=self.dtype),
            operator,
            IdentityOperator(self.shapein),
        )

    def _rule_add(self, s):
        return ScalarOperator(
            self.data + s.data,
            shapein=self.shapein or s.shapein,
            dtype=self._find_common_type([self.dtype, s.dtype]),
        )

    def _rule_mul(self, s):
        return ScalarOperator(
            self.data * s.data,
            shapein=self.shapein or s.shapein,
            dtype=self._find_common_type([self.dtype, s.dtype]),
        )


@real
@idempotent
@involutary
@inplace
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
        output[...] = input

    def toshapein(self, v):
        return v.reshape(self.shapein)

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

    def __init__(
        self, data, broadcast='disabled', shapein=None, dtype=None, **keywords
    ):
        if data is None:
            raise ValueError('The data array is None.')

        if dtype is None:
            data = np.asarray(data)
            dtype = data.dtype
        self.data = np.array(data, dtype, copy=False, order='c', ndmin=1)

        broadcast = broadcast.lower()
        values = ('fast', 'slow', 'disabled')
        if broadcast not in values:
            raise ValueError(
                "Invalid value '{0}' for the broadcast keyword. Ex"
                "pected values are {1}.".format(broadcast, strenum(values))
            )
        if broadcast == 'disabled':
            if shapein not in (None, self.data.shape):
                raise ValueError(
                    "The input shapein is incompatible with the da" "ta shape."
                )
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
                raise ValueError(
                    "The data array cannot be broadcast across the" " input."
                )
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
