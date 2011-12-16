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

from collections import namedtuple
from . import memory
from .utils import (first_is_not, isclassattr, isscalar, ndarraywrap,
                    strenum, strshape, tointtuple)
from .decorators import (flags, real, idempotent, involutary, square, symmetric,
                         inplace)

__all__ = [
    'Operator',
    'OperatorFlags',
    'AdditionOperator',
    'BlockColumnOperator',
    'BlockDiagonalOperator',
    'BlockRowOperator',
    'BroadcastingOperator',
    'CompositionOperator',
    'DiagonalOperator',
    'IdentityOperator',
    'MultiplicationOperator',
    'ReshapeOperator',
    'ScalarOperator',
    'ZeroOperator',
    'asoperator',
    'I',
    'O',
]

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
            if other_ == 'self':
                if not isinstance(other, self.reference.__class__):
                    return None
            elif other.__class__.__name__ != other_ and \
                 all(b.__name__ != other_ for b in other.__class__.__bases__):
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
                 shapeout=None, reshapein=None, reshapeout=None,
                 attrin={}, attrout={}, classin=None, classout=None,
                 dtype=None, flags=None):
            
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
        self._set_inout(shapein, shapeout, reshapein, reshapeout,
                        attrin, attrout, classin, classout)

        if isinstance(self.direct, (types.FunctionType, types.MethodType)):
            if isinstance(self.direct, types.MethodType):
                d = self.direct.im_func
            else:
                d = self.direct
            self.inplace_reduction = 'operation' in d.func_code.co_varnames

    attrin = {}
    attrout = {}
    classin = None
    classout = None
    shapein = None
    shapeout = None
    
    dtype = None
    flags = OperatorFlags(*9*(False,))
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

        if isinstance(input, Operator):
            return CompositionOperator([self, input])

        if self.direct is None:
            raise NotImplementedError('Call to ' + self.__name__ + ' is not imp'
                                      'lemented.')
        i, o = self._validate_input(input, output)
        with memory.manager(o):
            if not self.inplace and self.same_data(i, o):
                memory.up()
                o_ = memory.get(o.nbytes, o.shape, o.dtype, self.__name__) \
                           .view(o.dtype).reshape(o.shape)
            else:
                o_ = o
            self.direct(i, o_)
            if not self.inplace and self.same_data(i, o):
                memory.down()
                o[...] = o_

        cls = input.__class__ if isinstance(input, np.ndarray) else np.ndarray
        attr = input.__dict__.copy() if hasattr(input, '__dict__') else {}
        cls = self.propagate_attributes(cls, attr)
        if cls is np.ndarray and len(attr) > 0:
            cls = ndarraywrap
        if output is None:
            output = o
        if type(output) is np.ndarray:
            if cls is np.ndarray:
                return output
            output = output.view(cls)
        elif type(output) is not cls:
            output.__class__ = cls
            if output.__array_finalize__ is not None:
                output.__array_finalize__()

        # we cannot simply update __dict__, because of properties.
        # the iteration is sorted by key, so that attributes beginning with an
        # underscore are set first.
        for k in sorted(attr.keys()):
            setattr(output, k, attr[k])
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
        d = np.empty((n,m), self.dtype)
        v = np.zeros(n, self.dtype)
        for i in range(n):
            v[i] = 1
            o = d[i,:].reshape(shapeout)
            with memory.manager(o):
                self.direct(v.reshape(shapein), o)
            v[i] = 0
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

    def add_rule(self, subject, predicate, operation=None):
        """
        Add a rule to the rule list, taking care of duplicates.
        Rules matching classes have a lower priority than the others.
        """
        if operation is None:
            operation = CompositionOperator
        if issubclass(operation, CommutativeCompositeOperator) and \
           subject[-1] == '.':
            subject = '.' + subject[:-1]
        rule = OperatorRule(self, subject, predicate)
        if operation not in self.rules:
            self.rules[operation] = []
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

    def propagate_attributes(self, cls, attr):
        """
        Propagate attributes according to operator's output attributes.
        If the class changes, class attributes are removed if they are
        not class attributes of the new class.
        """
        if None not in (self.classout, cls) and self.classout is not cls:
            for a in attr.keys():
                if isclassattr(cls, a) and not isclassattr(self.classout, a):
                    del attr[a]
        if 'shape_global' in attr:
            del attr['shape_global']
        if isinstance(self.attrout, dict):
            attr.update(self.attrout)
        else:
            self.attrout(attr)
        return self.classout or cls
            
    def reshapein(self, shapein):
        """
        Return the operator's output shape.
        
        If the operator has not an explicit output shape, the specified input
        shape is used to deduce it.

        Parameter
        ---------
        shapein : tuple
            The input shape.

        Returns
        -------
        shapeout : tuple
            The output shape, consistent with the input shape
        """
        shapein = tointtuple(shapein)
        if None not in (self.shapein, shapein) and self.shapein != shapein:
            raise ValueError("The input shape '{0}' is incompatible with that o"
                "f {1}: '{2}'.".format(strshape(shapein), self.__name__,
                strshape(self.shapein)))
        if self.shapeout is not None:
            return self.shapeout
        if self._reshapein is not None:
            return tointtuple(self._reshapein(shapein))
        if self.flags.SQUARE:
            return shapein
        return None

    def reshapeout(self, shapeout):
        """
        Return the operator's input shape.
        
        If the operator has not an explicit input shape, the specified output
        shape is used to deduce it.

        Parameter
        ---------
        shapeout : tuple
            The output shape.

        Returns
        -------
        shapein : tuple
            The input shape, consistent with the output shape
        """
        shapeout = tointtuple(shapeout)
        if None not in (self.shapeout, shapeout)  and self.shapeout != shapeout:
            raise ValueError("The output shape '{0}' is incompatible with that "
                "of {1}: '{2}'.".format(strshape(shapeout), self.__name__,
                strshape(self.shapeout)))
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
                         self.shapein, reshapein=self._reshapeout, reshapeout= \
                         self._reshapein, attrin=self.attrout, attrout= \
                         self.attrin, classin=self.classout, classout= \
                         self.classin, dtype=self.dtype, flags=self.flags)
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
                         self._reshapein, attrin=self.attrout, attrout= \
                         self.attrin, classin=self.classout, classout= \
                         self.classin, dtype=self.dtype, flags=self.flags)
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
                         self._reshapein, attrin=self.attrout, attrout= \
                         self.attrin, classin=self.classout, classout= \
                         self.classin, dtype=self.dtype, flags=self.flags)
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
                          reshapeout=self._reshapein, attrin=self.attrout,
                          attrout=self.attrin, classin=self.classout, classout=\
                          self.classin, dtype=self.dtype, flags=self.flags)
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
                          reshapeout=self._reshapein, attrin=self.attrout,
                          attrout=self.attrin, classin=self.classout, classout=\
                          self.classin, dtype=self.dtype, flags=self.flags)
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
                          reshapeout=self._reshapein, attrin=self.attrout,
                          attrout=self.attrin, classin=self.classout, classout=\
                          self.classin, dtype=self.dtype, flags=self.flags)
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
        self.rules = {}
        if self.flags.IDEMPOTENT:
            self.add_rule('..', '.')
        if self.flags.ORTHOGONAL:
            self.add_rule('.T.', '1')
        if self.flags.UNITARY:
            self.add_rule('.H.', '1')
        if self.flags.INVOLUTARY:
            self.add_rule('..', '1')

    def _set_inout(self, shapein, shapeout, reshapein, reshapeout,
                   attrin, attrout, classin, classout):
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

        if not isinstance(attrin, dict) or not isinstance(attrout, dict):
            raise TypeError('Attributes should be given as a dictionary.')
        if len(attrin) > 0:
            self.attrin = attrin
        if len(attrout) > 0:
            self.attrout = attrout
        if classin is not None:
            self.classin = classin
        if classout is not None:
            self.classout = classout

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
                        strshape(shapeout), strshape(shapeout_)))
        elif shapein is not None and self._reshapeout is not None:
            shapein_ = tointtuple(self._reshapeout(shapeout))
            if shapein_ is not None and shapein_ != shapein:
                raise ValueError("The specified input shape '{0}' is incompati"
                        "ble with that given by reshapeout '{1}'.".format(
                        strshape(shapein), strshape(shapein_)))
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
        """
        Return the input and output as ndarray instances.
        If required, allocate the output.
        """
        input = np.array(input, copy=False, subok=True)

        shapeout = self.reshapein(input.shape)
        dtype = self._find_common_type([input.dtype, self.dtype])
        input = np.array(input, dtype=dtype, subok=False, copy=False)
        if output is not None:
            if output.dtype != dtype:
                raise ValueError("The output has an invalid dtype '{0}'. Expect"
                    "ed dtype is '{1}'.".format(output.dtype, dtype))
            if output.shape != shapeout:
                raise ValueError("The output has an invalid shape '{0}'. Expect"
                    "ed shape is '{1}'.".format(output.shape, shapeout))
            output = output.view(np.ndarray)
        else:
            output = memory.allocate(shapeout, dtype, None, self.__name__)[0]
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
                if isinstance(val, (dict, tuple, list)):
                    if val == defaults[ivar - nargs]:
                        continue
                elif val is defaults[ivar - nargs]:
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
            output[...] = operator.matvec(input)
        def transpose(input, output):
            output[...] = operator.rmatvec(input)
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

    def __init__(self, operands, *args, **keywords):
        dtype = self._find_common_type([o.dtype for o in self.operands])
        classin = first_is_not([o.classin for o in self.operands], None)
        classout = first_is_not([o.classout for o in self.operands], None)
        Operator.__init__(self, dtype=dtype, classin=classin, classout=classout,
                          **keywords)

    def propagate_attributes(self, cls, attr):
        return self.operands[0].propagate_attributes(cls, attr)
            
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
        elif isinstance(self, BlockDiagonalOperator):
            op = ' ⊕ '
        else:
            op = ' * '
        operands = ['({0})'.format(o) if isinstance(o, (AdditionOperator,
                    BlockDiagonalOperator)) else \
                    str(o) for o in self.operands]
        if isinstance(self, BlockDiagonalOperator):
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
        r += '\n    '+'\n    '.join(components) + '])'
        return r


class CommutativeCompositeOperator(CompositeOperator):
    """
    Class for commutative operator, such as addition or multiplication
    """
    def __new__(cls, operands, operation, *args, **keywords):
        return CompositeOperator.__new__(cls, operands, *args, **keywords)

    def __init__(self, operands, operation, *args, **keywords):
        CompositeOperator.__init__(self, operands, *args, **keywords)
        self.operation = operation

    def direct(self, input, output):
        operands = list(self.operands)
        assert len(operands) > 1

        try:
            ir = [o.inplace_reduction for o in operands]
            index = ir.index(False)
            operands[0], operands[index] = operands[index], operands[0]
            need_temporary = ir.count(False) > 1
        except ValueError:
            need_temporary = False

        if need_temporary:
            memory.up()
            buf = memory.get(output.nbytes, output.shape, output.dtype,
                      self.__name__).view(output.dtype).reshape(output.shape)

        operands[0].direct(input, output)

        for op in operands[1:]:
            if op.inplace_reduction:
                op.direct(input, output, operation=self.operation)
            else:
                op.direct(input, buf)
                self.operation(output, buf)

        if need_temporary:
            memory.down()

    def propagate_attributes(self, cls, attr):
        for op in self.operands:
            cls = op.propagate_attributes(cls, attr)
        return cls

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

    @classmethod
    def _apply_rules(cls, ops):
        if len(ops) <= 1:
            return ops
        i = 0
        while i < len(ops):
            if cls not in ops[i].rules:
                i += 1
                continue
            j = 0
            consumed = False
            while j < len(ops):
                if j != i:
                    for rule in ops[i].rules[cls]:
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


class AdditionOperator(CommutativeCompositeOperator):
    """
    Class for operator addition

    If at least one of the input already is the result of an addition,
    a flattened list of operators is created by associativity, to simplify
    reduction.
    """
    def __new__(cls, operands):
        return CommutativeCompositeOperator.__new__(cls, operands,
                                                    operator.__iadd__)

    def __init__(self, operands):
        flags = {
            'LINEAR':all([op.flags.LINEAR for op in self.operands]),
            'REAL':all([op.flags.REAL for op in self.operands]),
            'SQUARE':self.shapein is not None and \
                self.shapein == self.shapeout or \
                all([op.flags.SQUARE for op in self.operands])}
        CommutativeCompositeOperator.__init__(self, operands, operator.__iadd__,
                                              flags=flags)
        self.classin = first_is_not([o.classin for o in self.operands], None)
        self.classout = first_is_not([o.classout for o in self.operands], None)

    def associated_operators(self):
        return { 'T' : AdditionOperator([m.T for m in self.operands]),
                 'H' : AdditionOperator([m.H for m in self.operands]),
                 'C' : AdditionOperator([m.conjugate() for m in self.operands]),
               }
                

class MultiplicationOperator(CommutativeCompositeOperator):
    """
    Class for Hadamard (element-wise) multiplication of operators.

    If at least one of the input already is the result of an multiplication,
    a flattened list of operators is created by associativity, to simplify
    reduction.
    """
    def __new__(cls, operands):
        return CommutativeCompositeOperator.__new__(cls, operands,
                                                    operator.__imul__)

    def __init__(self, operands):
        flags = {
            'LINEAR':False,
            'REAL':all([op.flags.REAL for op in self.operands]),
            'SQUARE':self.shapein is not None and \
                self.shapein == self.shapeout or \
                all([op.flags.SQUARE for op in self.operands])}
        CommutativeCompositeOperator.__init__(self, operands, operator.__imul__,
                                              flags=flags)
        self.classin = first_is_not([o.classin for o in self.operands], None)
        self.classout = first_is_not([o.classout for o in self.operands], None)

    def associated_operators(self):
        return { 'C' : MultiplicationOperator([m.conjugate()
                                               for m in self.operands])}


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
            'LINEAR':all([op.flags.LINEAR for op in self.operands]),
            'REAL':all([op.flags.REAL for op in self.operands]),
            'SQUARE':self.shapein is not None and \
                (self.shapein == self.shapeout) or \
                all([op.flags.SQUARE for op in self.operands])}
        CompositeOperator.__init__(self, operands, flags=flags)
        self.classin = first_is_not([o.classin for o in reversed(
                                     self.operands)], None)
        self.classout = first_is_not([o.classout for o in self.operands], None)
        self.inplace_reduction = self.operands[0].inplace_reduction
        self._info = {}

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

    def direct(self, input, output, operation=None):

        inplace_composition = self.same_data(input, output)
        shapeouts, sizeouts, outplaces, reuse_output = self._get_info(
            input.shape, output.nbytes, output.dtype, inplace_composition and \
            operation is None)
        noutplaces = outplaces.count(True)

        nswaps = 0
        if not reuse_output:
            memory.up()
        elif inplace_composition and outplaces[-1] or \
             not inplace_composition and noutplaces % 2 == 0:
            memory.swap()
            nswaps += 1

        i = input
        for iop, (op, shapeout, sizeout, outplace) in enumerate(
            zip(self.operands, shapeouts, sizeouts, outplaces)[:0:-1]):
            if outplace and iop > 0:
                memory.up()
                o = memory.get(sizeout, shapeout, output.dtype, self.__name__) \
                          .view(output.dtype).reshape(shapeout)
                op.direct(i, o)
                i = o
                memory.down()
                memory.swap()
                nswaps += 1
            else:
                # we keep reusing the same stack element for inplace operators
                o = memory.get(sizeout, shapeout, output.dtype, self.__name__) \
                          .view(output.dtype).reshape(shapeout)
                op.direct(i, o)
                i = o

        if outplaces[0]:
            memory.up()
        if self.inplace_reduction:
            self.operands[0].direct(i, output, operation=operation)
        else:
            self.operands[0].direct(i, output)
        if outplaces[0]:
            memory.down()
            memory.swap()
            nswaps += 1
            
        if nswaps % 2 == 1:
            memory.swap()

        if not reuse_output:
            memory.down()

    def propagate_attributes(self, cls, attr):
        for op in reversed(self.operands):
            cls = op.propagate_attributes(cls, attr)
        return cls

    def reshapein(self, shape):
        for op in reversed(self.operands):
            shape = op.reshapein(shape)
        return shape

    def reshapeout(self, shape):
        for op in self.operands:
            shape = op.reshapeout(shape)
        return shape

    @classmethod
    def _apply_rules(cls, ops):
        if len(ops) <= 1:
            return ops
        i = len(ops) - 1

        while i >= 0:
            
            # inspect operators on the right
            consumed = False
            if i < len(ops) - 1 and cls in ops[i].rules:
                for rule in ops[i].rules[cls]:
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
            if i > 0 and cls in ops[i].rules:
                for rule in ops[i].rules[cls]:
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

    def _get_info(self, shape, nbytes, dtype, inplace):
        try:
            return self._info[(shape, nbytes, dtype, inplace)]
        except KeyError:
            pass
        shapeouts = self._get_shapeouts(shape)
        sizeouts = self._get_sizeouts(shapeouts)
        outplaces, reuse_output = self._get_outplaces(nbytes, inplace, sizeouts)
        v = shapeouts, sizeouts, outplaces, reuse_output
        self._info[(shape,nbytes,dtype,inplace)] = v
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
        if inplace_composition and noutplaces % 2 == 1 and \
           noutplaces == len(self.operands):
            return outplaces, False

        last_inplace_changed_to_outplace = False
        if inplace_composition:
            # if composition is inplace, enforce  even number of outplace
            if noutplaces % 2 == 1 and False in outplaces:
                index = outplaces.index(False)
                outplaces[index] = True
                last_inplace_changed_to_outplace = True
            output_is_requested = True # we start with the input=output
        else:
            output_is_requested = noutplaces % 2 == 0

        reuse_output = False
        for op, outplace, nbytes in zip(self.operands, outplaces,
                                        sizeouts)[:0:-1]:
            if outplace:
                output_is_requested = not output_is_requested
            if output_is_requested:
                if nbytes > output_nbytes:
                    if last_inplace_changed_to_outplace:
                        outplaces[index] = False # revert back
                    return outplaces, False
                reuse_output = True
        return outplaces, reuse_output

class BlockOperator(CompositeOperator):
    """
    Abstract base class for BlockDiagonalOperator, BlockColumnOperator and
    BlockRowOperator.
    """

    def __init__(self, operands, partitionin=None, partitionout=None,
                 axisin=None, axisout=None, new_axisin=None, new_axisout=None):

        if new_axisin is not None:
            if partitionin is None:
                partitionin = len(operands) * (1,)
            elif partitionin != len(operands) * (1,):
                raise ValueError('If the block operator input shape has one mor'
                                 'e dimension than its blocks, the input partit'
                                 'ion must be a tuple of ones.')
        if new_axisout is not None:
            if partitionout is None:
                partitionout = len(operands) * (1,)
            elif partitionout != len(operands) * (1,):
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
        if partitionout is not None:
            if len(partitionout) != len(operands):
                raise ValueError('The number of operators must be the same as t'
                                 'he length of the output partition.')
        flags = {
            'LINEAR':all([op.flags.LINEAR for op in self.operands]),
            'REAL':all([op.flags.REAL for op in self.operands])}

        if partitionin is not None and partitionout is not None:
            flags['SQUARE'] = all([op.flags.SQUARE for op in self.operands])

        self.partitionin = partitionin
        self.partitionout = partitionout
        self.axisin = axisin
        self.new_axisin = new_axisin
        self.axisout = axisout
        self.new_axisout = new_axisout
        self.slicein = self._get_slice(axisin, new_axisin)
        self.sliceout = self._get_slice(axisout, new_axisout)
        if partitionin is None:
            self.__class__ = BlockColumnOperator
        elif partitionout is None:
            self.__class__ = BlockRowOperator
        else:
            self.__class__ = BlockDiagonalOperator
        CompositeOperator.__init__(self, operands, flags=flags)
        self.add_rule('.{Operator}', self._rule_operator_add, AdditionOperator)
        self.add_rule('.{self}', self._rule_add, AdditionOperator)
        self.add_rule('.{Operator}', self._rule_operator_comp_right)
        self.add_rule('{Operator}.', self._rule_operator_comp_left)
        self.add_rule('.{BlockOperator}', self._rule_comp_right)
        self.add_rule('{BlockOperator}.', self._rule_comp_left)

    def reshapein(self, shapein):
        if shapein is None or self.partitionin is None:
            shapeouts = [op.reshapein(shapein) for op in self.operands]
        else:
            shapeouts = [op.reshapein(s) for op,s in zip(self.operands,
                self._get_shapes(shapein, self.partitionin, self.axisin,
                self.new_axisin))]
        if None in shapeouts and shapein is not None:
            raise ValueError("The 'reshapein' method of implicit-shape operator"
                "s must not return None for an explicit input shape.")
        shapeout = self._validate_shapes(shapeouts, self.partitionout,
                                         self.axisout, self.new_axisout)
        if shapein is None:
            if shapeout is None or None in shapeouts:
                return None
        if self.partitionout is None:
            return shapeout
        if self.new_axisout is not None:
            a = self.new_axisout
            if self.new_axisout < 0:
                a += len(shapeout) + 1
            return shapeout[:a] + (len(self.operands),) + shapeout[a:]
        shapeout = list(shapeout)
        shapeout[self.axisout] = sum([s[self.axisout] for s in shapeouts])
        return tointtuple(shapeout)

    def reshapeout(self, shapeout):
        if shapeout is None or self.partitionout is None:
            shapeins = [op.reshapeout(shapeout) for op in self.operands]
        else:
            shapeins = [op.reshapeout(s) for op,s in zip(self.operands,
                self._get_shapes(shapeout, self.partitionout, self.axisout,
                self.new_axisout))]
        if None in shapeins and shapeout is not None:
            raise ValueError("The 'reshapeout' method of implicit-shape operato"
                "rs must not return None for an explicit output shape.")
        shapein = self._validate_shapes(shapeins, self.partitionin,
                                         self.axisin, self.new_axisin)
        if shapeout is None:
            if shapein is None or None in shapeins:
                return None
        if self.partitionin is None:
            return shapein
        if self.new_axisin is not None:
            a = self.new_axisin
            if self.new_axisin < 0:
                a += len(shapein) + 1
            return shapein[:a] + (len(self.operands),) + shapein[a:]
        shapein = list(shapein)
        shapein[self.axisin] = sum([s[self.axisin] for s in shapeins])
        return tointtuple(shapein)

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
                    shapein = op.reshapeout(shapeout)
                    pin.append(shapein[axisin])
                except IndexError:
                    continue
            if len(pin) == 0 or any([p != pin[0] for p in pin]):
                continue
            partitionin[i] = pin[0]
        return tuple(partitionin)

    @staticmethod
    def _get_partitionout(ops, partitionin, axisin, axisout, new_axisin,
                          new_axisout):
        """ Infer the output partition from the input partition. """
        if new_axisout is not None:
            return len(ops) * (1,)
        
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
                    shapeout = op.reshapein(shapein)
                    pout.append(shapeout[axisout])
                except IndexError:
                    continue
            print pout
            if len(pout) == 0 or any([p != pout[0] for p in pout]):
                continue
            partitionout[i] = pout[0]
        return tuple(partitionout)

    @staticmethod
    def _get_shapes(shape, partition, axis, new_axis):
        if None in partition:
            raise ValueError('The shape of an operator with implicit partition '
                             'cannot be inferred.')
        if new_axis is not None:
            shape_ = list(shape)
            del shape_[new_axis]
            shapes = len(partition) * (tuple(shape_),)
            print 'shapes', shapes
            return shapes
        shapes = []
        for p in partition:
            shape_ = list(shape)
            shape_[axis] = p
            shapes.append(shape_)
        return tuple(shapes)

    @staticmethod
    def _get_slice(axis, new_axis):
        """ Compute the tuple of slices to extract a block from the input. """
        axis = axis if axis is not None else new_axis
        if axis is None:
            return None
        if axis >= 0:
            return (axis+1) * [slice(None)] + [Ellipsis]
        return [Ellipsis] + (-axis) * [slice(None)]

    @staticmethod
    def _validate_composition(op1, op2):
        axisin1 = op1.axisin if op1.axisin is not None else op1.new_axisin
        axisout2 = op2.axisout if op2.axisout is not None else op2.new_axisout
        if axisin1 != axisout2:
            return None
        if op1.axisin is not None and op2.new_axisout is not None or \
           op1.new_axisin is not None and op2.axisout is not None:
            # we could handle these cases with a reshape
            return None
        p1 = op1.partitionin
        p2 = op2.partitionout
        if p1 is None or p2 is None or len(p1) != len(p2):
            return None
        if any(p != q for p, q in zip(p1, p2) if None not in (p,q)):
            return None
        return op2.partitionin, op1.partitionout

    @staticmethod
    def _validate_addition(op1, op2):
        axisin1 = op1.axisin if op1.axisin is not None else op1.new_axisin
        axisin2 = op2.axisin if op2.axisin is not None else op2.new_axisin
        axisout1 = op1.axisout if op1.axisout is not None else op1.new_axisout
        axisout2 = op2.axisout if op2.axisout is not None else op2.new_axisout
        if axisin1 != axisin2 or axisout1 != axisout2:
            return None, None
        if op1.axisin is not None and op2.new_axisin is not None or \
           op1.new_axisin is not None and op2.axisin is not None or \
           op1.axisout is not None and op2.new_axisout is not None or \
           op1.new_axisout is not None and op2.axisout is not None:
            # we could handle these cases with a reshape
            return None
        def func(p1, p2):
            if p1 is None and p2 is not None or p1 is not None and p2 is None:
                return None
            if len(p1) != len(p2):
                return None
            if any(p != q for p, q in zip(p1, p2) if None not in (p,q)):
                return None
            return [ p or q for p, q in zip(p1, p2)]
        return func(op1.partitionin, op2.partitionin), \
               func(op1.partitionout, op2.partitionout)

    @staticmethod
    def _validate_shapes(shapes, p, axis, new_axis):
        explicit = [s for s in shapes if s is not None]
        if len(explicit) == 0:
            return None
        shape = explicit[0]
        if p is None or new_axis is not None:
            if any([s != shape for s in explicit]):
                raise ValueError("The operands have incompatible shapes: '{0}'"
                                 ".".format(strshape(shapes)))
            return shape
        rank = len(shape)
        if any([len(s) != rank for s in explicit]):
            raise ValueError("The blocks do not have the same number of dimensi"
                "ons: '{0}'.".format(strshape(shapes)))
        if any([shapes[i] is not None and shapes[i][axis] != p[i] \
                for i in range(len(p)) if p[i] is not None]):
            raise ValueError("The blocks have shapes '{0}' incompatible with th"
                "e partition {1}.".format(strshape(shapes), strshape(p)))
        if len(explicit) == 1:
            return shape
        ok = [all([s is None or s[i] == shape[i] for s in shapes]) \
              for i in range(rank)]
        ok[axis] = True
        if not all(ok):
            raise ValueError("The dimensions of the blocks '{0}' are not the sa"
                "me along axes other than that of the partition '{1}'.".format(
                strshape(shapes), strshape(p)))
        return shape

    def _rule_operator_add(self, op):
        """ Rule for BlockOperator + Operator. """
        if op.shapein is not None:
            return None
        return BlockOperator([o + op for o in self.operands],
            self.partitionin, self.partitionout, self.axisin, self.axisout,
            self.new_axisin, self.new_axisout)

    def _rule_add(self, p):
        """ Rule for BlockOperator + BlockOperator. """
        partitionin, partitionout = self._validate_addition(p, self)
        if partitionin is partitionout is None:
            return None
        operands = [o1 + o2 for o1,o2 in zip(p.operands, self.operands)]
        return BlockOperator(operands, partitionin, partitionout,
            self.axisin, self.axisout, self.new_axisin, self.new_axisout)

    def _rule_operator_comp_left(self, op):
        """ Rule for Operator * BlockOperator. """
        if self.partitionout is None:
            return None
        if op.shapeout is not None:
            return None
        n = len(self.partitionout)
        partitionout = self._get_partitionout(n*[op], self.partitionout,
            self.axisout, self.axisout, self.new_axisout, self.new_axisout)
        return BlockOperator([op * o for o in self.operands],
            self.partitionin, partitionout, self.axisin, self.axisout,
            self.new_axisin, self.new_axisout)

    def _rule_operator_comp_right(self, op):
        """ Rule for BlockOperator * Operator. """
        if self.partitionin is None:
            return None
        if op.shapein is not None:
            return None
        n = len(self.partitionin)
        partitionin = self._get_partitionin(n*[op], self.partitionin,
            self.axisin, self.axisin, self.new_axisin, self.new_axisin)
        return BlockOperator([o * op for o in self.operands],
            partitionin, self.partitionout, self.axisin, self.axisout,
            self.new_axisin, self.new_axisout)

    def _rule_comp_left(self, p):
        return self._rule_comp(p, self)

    def _rule_comp_right(self, p):
        return self._rule_comp(self, p)

    def _rule_comp(self, p1, p2):
        """ Rule for BlockOperator * BlockOperator. """
        partitions = self._validate_composition(p1, p2)
        if partitions is None:
            return None
        partitionin, partitionout = partitions
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
    p = BlockDiagonalOperator([o1, o2], axis=-1)
    print(p.shapein)
    (16,7)

    """
    def __init__(self, operands, partitionin=None, axisin=None, axisout=None,
                 new_axisin=None, new_axisout=None):
   
        if axisin is None and new_axisin is None:
            new_axisin = 0
        if axisout is None:
            axisout = axisin
        if new_axisout is None:
            new_axisout = new_axisin

        if partitionin is None:
            partitionin = self._get_partition([op.shapein for op in operands],
                                              axisin, new_axisin)
        partitionin = tointtuple(partitionin)
        partitionout = self._get_partitionout(operands, partitionin, axisin,
                                              axisout, new_axisin, new_axisout)

        BlockOperator.__init__(self, operands, partitionin, partitionout,
                               axisin, axisout, new_axisin, new_axisout)

    def associated_operators(self):
        return {
            'C': BlockDiagonalOperator([op.C for op in self.operands],
                     self.partitionin, self.axisin, self.axisout,
                     self.new_axisin, self.new_axisout),
            'T': BlockDiagonalOperator([op.T for op in self.operands],
                     self.partitionout, self.axisout, self.axisin,
                     self.new_axisout, self.new_axisin),
            'H': BlockDiagonalOperator([op.H for op in self.operands],
                     self.partitionout, self.axisout, self.axisin,
                     self.new_axisout, self.new_axisin),
            'I': BlockDiagonalOperator([op.I for op in self.operands],
                     self.partitionout, self.axisout, self.axisin,
                     self.new_axisout, self.new_axisin)}

    def direct(self, input, output):
        if None in self.partitionout:
            shapeins = self._get_shapes(input.shape, self.partitionin,
                                        self.axisin, self.new_axisin)
            partitionout = [op.reshapein(s)[self.axisout] \
                            for op,s in zip(self.operands, shapeins)]
        else:
            partitionout = self.partitionout
        destin = 0
        destout = 0
        for op, nin, nout in zip(self.operands, self.partitionin, partitionout):
            if self.new_axisin is not None:
                self.slicein[self.new_axisin] = destin
            else:
                self.slicein[self.axisin] = slice(destin, destin + nin)
            if self.new_axisout is not None:
                self.sliceout[self.new_axisout] = destout
            else:
                self.sliceout[self.axisout] = slice(destout, destout + nout)
            op.direct(input[self.slicein], output[self.sliceout])
            destin += nin
            destout += nout


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
                 new_axisout=None):
        if axisout is None and new_axisout is None:
            new_axisout = 0
        if partitionout is None:
            partitionout = self._get_partition([op.shapeout for op in operands],
                                               axisout, new_axisout)
        partitionout = tointtuple(partitionout)

        BlockOperator.__init__(self, operands, partitionout=partitionout,
                               axisout=axisout, new_axisout=new_axisout)

    def associated_operators(self):
        return {
            'C': BlockColumnOperator([op.C for op in self.operands],
                     self.partitionout, self.axisout, self.new_axisout),
            'T': BlockRowOperator([op.T for op in self.operands],
                     self.partitionout, self.axisout, self.new_axisout),
            'H': BlockRowOperator([op.H for op in self.operands],
                     self.partitionout, self.axisout, self.new_axisout)}
        
    def direct(self, input, output):
        if None in self.partitionout:
            raise NotImplementedError()
        dest = 0
        for op, n in zip(self.operands, self.partitionout):
            if self.new_axisout is not None:
                self.sliceout[self.new_axisout] = dest
            else:
                self.sliceout[self.axisout] = slice(dest, dest + n)
            op.direct(input, output[self.sliceout])
            dest += n

    def __str__(self):
        operands = ['[{}]'.format(o) for o in self.operands]
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
    def __init__(self, operands, partitionin=None, axisin=None,
                 new_axisin=None):
        if axisin is None and new_axisin is None:
            new_axisin = 0
        if partitionin is None:
            partitionin = self._get_partition([op.shapein for op in operands],
                                              axisin, new_axisin)
        partitionin = tointtuple(partitionin)

        BlockOperator.__init__(self, operands, partitionin=partitionin,
                               axisin=axisin, new_axisin=new_axisin)

    def associated_operators(self):
        return {
            'C': BlockRowOperator([op.C for op in self.operands],
                     self.partitionin, self.axisin, self.new_axisin),
            'T': BlockColumnOperator([op.T for op in self.operands],
                     self.partitionin, self.axisin, self.new_axisin),
            'H': BlockColumnOperator([op.H for op in self.operands],
                     self.partitionin, self.axisin, self.new_axisin)}

    def direct(self, input, output):
        if None in self.partitionin:
            raise NotImplementedError()
        work = np.zeros_like(output)
        dest = 0
        for op, n in zip(self.operands, self.partitionin):
            if self.new_axisin is not None:
                self.slicein[self.new_axisin] = dest
            else:
                self.slicein[self.axisin] = slice(dest, dest + n)
            op.direct(input[self.slicein], output)
            work += output
            dest += n
        output[...] = work

    def __str__(self):
        operands = [str(o) for o in self.operands]
        if len(operands) > 2:
            operands = [operands[0], '...', operands[-1]]
        return '[[ ' + ' '.join(operands) + ' ]]'


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
        self.add_rule('.T.', '1')

    def direct(self, input, output):
        if self.same_data(input, output):
            pass
        output.ravel()[:] = input.ravel()

    def associated_operators(self):
        return {'T':ReshapeOperator(self.shapeout, self.shapein)}

    def __str__(self):
        return strshape(self.shapeout) + '←' + strshape(self.shapein)


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


@symmetric
@inplace
class DiagonalOperator(BroadcastingOperator):

    def __new__(cls, data, broadcast='disabled', shapein=None, dtype=None):
        """
        Subclass of BroadcastingOperator.

        Arguments
        ---------

        data : ndarray
          The diagonal coefficients

        broadcast : 'fast' or 'disabled' (default 'disabled')
          If broadcast == 'fast', the diagonal is broadcasted along the fast
          axis.

        Exemple
        -------
        >>> A = DiagonalOperator(arange(1, 6, 2))
        >>> A.todense()

        array([[1, 0, 0],
               [0, 3, 0],
               [0, 0, 5]])

        >>> A = DiagonalOperator(arange(1, 3), broadcast='fast', shapein=(2, 2))
        >>> A.todense()

        array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 2, 0],
               [0, 0, 0, 2]])
        """

        data = np.array(data, dtype, copy=False)
        if shapein is None and broadcast == 'disabled' and data.ndim > 0:
            shapein = data.shape
        if np.all(data == 1):
            return IdentityOperator(shapein, dtype)
        elif np.all(data == 0):
            return ZeroOperator(shapein, dtype)
        return BroadcastingOperator.__new__(cls, data, broadcast=broadcast,
            shapein=shapein, dtype=dtype)

    def __init__(self, data, broadcast='disabled', shapein=None, dtype=None):
        BroadcastingOperator.__init__(self, data, broadcast, shapein, dtype)
        self.add_rule('{DiagonalOperator}.',
                      lambda o: self._rule_diagonal(o,np.add), AdditionOperator)
        self.add_rule('{DiagonalOperator}.',
                      lambda o: self._rule_diagonal(o, np.multiply))
        self.add_rule('{ScalarOperator}.',
                      lambda o: self._rule_scalar(o, np.add), AdditionOperator)
        self.add_rule('{ScalarOperator}.',
                      lambda o: self._rule_scalar(o, np.multiply))

    def direct(self, input, output):
        if self.broadcast == 'fast':
            np.multiply(input.T, self.data.T, output.T)
        else:
            np.multiply(input, self.data, output)

    def conjugate_(self, input, output):
        if self.broadcast == 'fast':
            np.multiply(input.T, np.conjugate(self.data).T, output.T)
        else:
            np.multiply(input, np.conjugate(self.data), output)

    def inverse(self, input, output):
        if self.broadcast == 'fast':
            np.divide(input.T, self.data.T, output.T)
        else:
            np.divide(input, self.data, output)

    def inverse_conjugate(self, input, output):
        if self.broadcast == 'fast':
            np.divide(input.T, np.conjugate(self.data).T, output.T)
        else:
            np.divide(input, np.conjugate(self.data), output)

    def _rule_scalar(self, s, operation):
        return DiagonalOperator(operation(s.data, self.data), self.broadcast,
            shapein=self.shapein, dtype=self._find_common_type([self.dtype,
            s.dtype]))

    def _rule_diagonal(self, d, operation):
        if set([self.broadcast, d.broadcast]) == set(['fast', 'slow']):
            raise ValueError('Fast and slow broadcast cannot be combined.')
        if self.broadcast == d.broadcast:
            broadcast = self.broadcast
        else:
            broadcast = 'disabled'
        if self.broadcast == 'fast' or d.broadcast == 'fast':
            data = (operation(self.data.T, d.data.T)).T
        else:
            data = operation(self.data, d.data)
        return DiagonalOperator(data, broadcast, self.shapein or d.shapein,
            dtype=self._find_common_type([self.dtype, d.dtype]))


@symmetric
@inplace
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
        self.add_rule('{ScalarOperator}.', self._rule_add, AdditionOperator)

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


@real
@flags(SQUARE=False, SYMMETRIC=False)
@inplace
class ZeroOperator(ScalarOperator):
    """
    A subclass of ScalarOperator with value=0.
    All __init__ keyword arguments are passed to the
    ScalarOperator __init__.
    """
    def __init__(self, shapein=None, shapeout=None, dtype=None, reshapein=None,
                 reshapeout=None):
        if shapein is not None and shapein == shapeout or reshapein is None and\
           reshapeout is None:
            flags = {'SQUARE':True, 'SYMMETRIC': True, 'IDEMPOTENT': True}
        else:
            flags = None
        Operator.__init__(self, shapein=shapein, shapeout=shapeout, reshapein= \
            reshapein, reshapeout=reshapeout, dtype=dtype, flags=flags)
        self.data = np.array(0)
        self.add_rule('.{Operator}', self._rule_zero_times)
        self.add_rule('{Operator}.', self._rule_times_zero)

    def associated_operators(self):
        return {
            'T' : ZeroOperator(shapein=self.shapeout, shapeout=self.shapein,
                      reshapein=self._reshapeout, reshapeout=self._reshapein,
                      dtype=self.dtype)
        }

    def direct(self, input, output):
        output[...] = 0

    def _combine_operators(self, o1, o2):
        result = ZeroOperator(shapein=o2.shapein or o2.reshapeout(o1.shapein),
            shapeout=o1.shapeout or o2.shapeout,
            reshapein=lambda s: o1.reshapein(o2.reshapein(s)),
            reshapeout=lambda s: o2.reshapeout(o1.reshapeout(s)),
            dtype=self._find_common_type([o1.dtype, o2.dtype]))
        result.toshapein = lambda v: o2.toshapein(v)
        return result

    def _rule_zero_times(self, op):
        return self._combine_operators(self, op)

    def _rule_times_zero(self, op):
        if not op.flags.LINEAR:
            return None
        return self._combine_operators(op, self)
        
        
I = IdentityOperator()
O = ZeroOperator()
