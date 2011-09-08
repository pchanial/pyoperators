from __future__ import division

import copy
import gc
import numpy as np
import scipy.sparse.linalg

from collections import namedtuple
from .utils import isscalar, strenum
from .decorators import square, symmetric

__all__ = [
    'Operator',
    'OperatorFlags',
    'AdditionOperator',
    'CompositionOperator',
    'ScalarOperator',
    'BroadcastingOperator',
    'asoperator',
]

verbose = True

class OperatorFlags(namedtuple('OperatorFlags',
                               ['LINEAR',
                                'SQUARE',
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
                 shapeout=None, dtype=None, flags=None):
            
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

        self._generated = False
        self._C = self._T = self._H = self._I = None

        self._set_dtype(dtype)
        self._set_flags(self, flags)
        self._set_inout(shapein, shapeout)
        self._set_name()

    flags = OperatorFlags(*9*(False,))
    shapein = None
    shapeout = None
    dtype = None

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

    def reshapein(self, shapein):
        """For explicit-shape operators, return operator's output shape.
        Otherwise, compute it from a given input shape."""
        if self.shapeout is not None:
            return self.shapeout
        return shapein

    def reshapeout(self, shapeout):
        """For explicit-shape operators, return operator's input shape.
        Otherwise, compute it from a given output shape."""
        if self.shapein is not None:
            return self.shapein
        return shapeout

    @staticmethod
    def same_data(array1, array2):
        return array1.__array_interface__['data'][0] == \
               array2.__array_interface__['data'][0]

    def todense(self, shapein=None):
        if not self.flags.LINEAR:
            raise TypeError('The operator is not linear.')
        shapein = shapein or self.shapein
        if shapein is None:
            raise ValueError("The operator has an implicit shape. Use the 'shap"
                             "pin' keyword.")
        shapeout = self.reshapein(shapein)
        m, n = np.product(shapeout), np.product(shapein)
        d = np.empty((n,m), self.dtype)
        v = np.zeros(n, self.dtype)
        for i in range(n):
            v[i] = 1
            self.direct(v.reshape(shapein), d[i,:].reshape(shapeout))
            v[i] = 0
        return d.T

    def matvec(self, v):
        v = self.toshapein(v)
        input, output = self._validate_input(v, None)
        self.direct(input, output)
        return output.ravel()

    def rmatvec(self, v):
        return self.T.matvec(v)

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
        if not self._generated:
            self._generate_associated_operators()
        return self._C

    @property
    def T(self):
        """Return the transpose of the operator."""
        if not self._generated:
            self._generate_associated_operators()
        return self._T

    @property
    def H(self):
        """Return the adjoint of the operator."""
        if not self._generated:
            self._generate_associated_operators()
        return self._H

    @property
    def I(self):
        """Return the inverse of the operator."""
        if not self._generated:
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
            C = Operator(self.conjugate_, dtype=self.dtype, flags=self.flags)
            C.__name__ = self.__name__ + '.C'

        if self.flags.SYMMETRIC:
            T = self
        elif 'T' in ops:
            T = ops['T']
        else:
            T = Operator(self.transpose, dtype=self.dtype, flags=self.flags)
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
            H = Operator(self.adjoint, dtype=self.dtype, flags=self.flags)
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
            I = Operator(self.inverse, dtype=self.dtype, flags=self.flags)
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
            IC = Operator(self.inverse_conjugate, dtype=self.dtype,
                          flags=self.flags)
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
            IT = Operator(self.inverse_transpose, dtype=self.dtype,
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
            IH = Operator(self.inverse_adjoint, dtype=self.dtype,
                          flags=self.flags)
            IH.__name__ = self.__name__ + '.I.H'

        for op in (T, H, I, IC):
            op.shapein, op.shapeout = self.shapeout, self.shapein
            op.toshapein, op.toshapeout = self.toshapeout, self.toshapein
            op.reshapein, op.reshapeout = self.reshapeout, self.reshapein
        
        for op in (C, IT, IH):
            op.shapein = self.shapein
            op.shapeout = self.shapeout

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

        for op in (self, C, T, H, I, IC, IT, IH):
            op._generated = True

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

    def _set_inout(self, shapein, shapeout):
        """Set methods and attributes dealing with the input and output
        handling."""
        if shapein is not None:
            if isscalar(shapein):
                shapein = (shapein,)
            self.shapein = tuple(int(s) for s in shapein)

        if self.flags.SQUARE:
            self.shapeout = self.shapein
            self.reshapeout = self.reshapein
            self.toshapeout = self.toshapein
        else:
            if shapeout is None:
                shapeout = self.reshapein(shapein)
            elif isscalar(shapeout):
                shapeout = (shapeout,)
            if shapeout is not None:
                self.shapeout = tuple(int(s) for s in shapeout)
                    
    def _set_name(self):
        """Set operator's __name__ attribute."""
        if self.__class__ != 'Operator':
            self.__name__ = self.__class__.__name__
        elif self.direct is not None:
            self.__name__ = self.direct.__name__
            if self.__name__ in ('<lambda>', 'direct'):
                self.__name__ = 'Operator'                

    def _validate_input(self, input, output):
        """Return the input as ndarray subclass and allocate the output
        if necessary."""
        input = np.array(input, copy=False, subok=True, ndmin=1)
        if type(input) is np.ndarray:
            input = input.view(ndarraywrap)

        if self.shapein is not None and self.shapein != input.shape:
            raise ValueError('The input of {0} has an invalid shape {1}. Expect'
                'ed shape is {2}.'.format(self.__name__,
                input.shape, self.shapein))
        shapeout = self.reshapein(input.shape)
        dtype = _get_dtypeout(input.dtype, self.dtype)
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
        result = self.__name__
        if self.shapein is not None or self.shapeout is not None:
            result += ' [input:'
            if self.shapein is None:
                result += 'unconstrained'
            else:
                result += str(self.shapein).replace(' ','')
            result += ', output:'
            if self.shapeout is None:
                result += 'unconstrained'
            else:
                result += str(self.shapeout).replace(' ','')
            result += ']'
        return result


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
                        dtype=operator.dtype)
    
    if isscalar(operator):
        return ScalarOperator(operator)

    return asoperator(scipy.sparse.linalg.aslinearoperator(operator))


class CompositeOperator(Operator):
    """
    Class for grouping operands
    """

    def __new__(cls, operands):
        operands = cls._validate_operands(operands)
        operands = cls._reduce_commute_scalar(operands)
        if len(operands) == 1:
            return operands[0]
        instance = super(CompositeOperator, cls).__new__(cls)
        instance.operands = operands
        return instance

    @property
    def dtype(self):
        return max([b.dtype for b in self.operands])

    @dtype.setter
    def dtype(self, dtype):
        pass

    @classmethod
    def _reduce_commute_scalar(cls, ops):
        if issubclass(cls, AdditionOperator):
            opn = np.add
        elif issubclass(cls, CompositionOperator):
            opn = np.multiply
        else:
            return ops

        # moving scalars from right to left
        if len(ops) < 2:
            return ops
        ops = list(ops)
        i = len(ops) - 2
        while i >= 0:
            if isinstance(ops[i+1], ScalarOperator) and not ops[i+1].shapein:
                if isinstance(ops[i], ScalarOperator):
                    ops[i] = ScalarOperator(opn(ops[i].data, ops[i+1].data))
                    del ops[i+1]
                elif ops[i].flags.LINEAR:
                    ops[i], ops[i+1] = ops[i+1], ops[i]
                elif opn == np.multiply:
                    if ops[i+1].data == 1:
                        del ops[i+1]
            i -= 1
        if len(ops) > 1 and opn == np.multiply and \
           isinstance(ops[0], ScalarOperator) and ops[0].data == 1 and \
           ops[0].shapein is None:
            del ops[0]

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
        result = Operator.__str__(self) + ':'
        components = []
        for operand in self.operands:
            components.extend(str(operand).split('\n'))
        result += '\n    '+'\n    '.join(components)
        return result


class AdditionOperator(CompositeOperator):
    """
    Class for operator addition

    If at least one of the input already is the result of an addition,
    a flattened list of operators is created by associativity, in order to
    benefit from the Operator's caching mechanism.
    """
    def __init__(self, operands):
        flags = {
            'LINEAR':all([op.flags.REAL for op in self.operands]),
            'REAL':all([op.flags.REAL for op in self.operands]),
            'SQUARE':self.shapein is not None and \
                (self.shapein == self.shapeout) or \
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

    @property
    def shapein(self):
        shapein = None
        for op in self.operands:
            shapein_ = op.shapein
            if shapein_ is None:
                continue
            if shapein is None or type(shapein_[-1]) is tuple:
                shapein = shapein_
                continue
            if shapein != shapein_:
                raise ValueError("Incompatible shape in operands: '" + \
                    str(shapein) +"' and '" + str(shapein_) + "'.")
        return shapein

    @shapein.setter
    def shapein(self, value):
        pass

    @property
    def shapeout(self):
        shapeout = None
        for op in self.operands:
            shapeout_ = op.shapeout
            if shapeout_ is None:
                continue
            if shapeout is None or type(shapeout_[-1]) is tuple:
                shapeout = shapeout_
                continue
            if shapeout != shapeout_:
                raise ValueError("Incompatible shape in operands: '" + \
                    str(shapeout) +"' and '" + str(shapeout_) + "'.")
        return shapeout

    @shapeout.setter
    def shapeout(self, value):
        pass


class CompositionOperator(CompositeOperator):
    """
    Class for acquisition models composition

    If at least one of the input already is the result of a composition,
    a flattened list of operators is created by associativity, in order to
    benefit from the Operator's caching mechanism.
    """

    def __init__(self, operands):
        flags = {
            'LINEAR':all([op.flags.REAL for op in self.operands]),
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
            o = self._get_output(op.reshapein(input.shape), input.dtype)
            op._propagate(output, o)
            op.direct(i, o)
            output.__class__ = o.__class__
            i = o
            print 'direct comp', output.__class__, output.__dict__

        # remove output from the work pool, to avoid side effects on the output
        self._del_output()

    @property
    def shapein(self):
        shape = None
        for model in self.operands:
            shape = model.reshapeout(shape)
        return shape

    @shapein.setter
    def shapein(self, value):
        pass

    @property
    def shapeout(self):
        shape = None
        for model in reversed(self.operands):
            shape = model.reshapein(shape)
        return shape

    @shapeout.setter
    def shapeout(self, value):
        pass

    def _get_output(self, shape, dtype):
        nbytes = np.product(shape) * dtype.itemsize
        if nbytes <= self.work[0].nbytes:
            return self.work[0][:nbytes].view(dtype).reshape(shape)

        buf, new = self._allocate(nbytes, np.int8, self.work[1])
        if new:
            self.work[1] = buf
        return buf

    def _set_output(self, output):
        self.work[0] = output.ravel().view(np.int8)
        
    def _del_output(self):
        self.work[0] = None


@symmetric
class ScalarOperator(Operator):
    """
    Multiplication by a scalar.

    """
    def __init__(self, value, shapein=None, dtype=None):
        value = np.asarray(value)
        if dtype is None:
            dtype = np.find_common_type([value.dtype, float], [])
            value = np.array(value, dtype=dtype)
        self.data = value
        
        if value in (1, -1):
            flags = {'IDEMPOTENT':True, 'INVOLUTARY':True}
        else:
            flags = None

        Operator.__init__(self, lambda i,o: np.multiply(i, value, o),
                          shapein=shapein, dtype=dtype, flags=flags)

    def associated_operators(self):
        return {
            'C' : ScalarOperator(np.conjugate(self.data), shapein=self.shapein,
                                 dtype=self.dtype),
            'I' : ScalarOperator(1/self.data, shapein=self.shapein,
                                 dtype=self.dtype),
            'IC' : ScalarOperator(np.conjugate(1/self.data),
                                  shapein=self.shapein, dtype=self.dtype)
        }

    def __str__(self):
        value = self.data.flat[0]
        if value == int(value):
            value = int(value)
        return str(value)


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
            if shapein not in (None, data.shape):
                raise ValueError("The input shapein is incompatible with the da"
                                 "ta shape.")
            shapein = data.shape
        self.broadcast = broadcast

        Operator.__init__(self, shapein=shapein, dtype=dtype, **keywords)

    def reshapein(self, shape):
        if self.shapeout is not None:
            return self.shapeout
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


def _get_dtypeout(d1, d2):
    """Return dtype of greater type rank."""
    if d1 is None:
        return d2
    if d2 is None:
        return d1
    return np.find_common_type([d1, d2], [])
