from __future__ import division

import copy
import gc
import numpy as np
import scipy.sparse.linalg

from collections import namedtuple

__all__ = [
    'Operator',
    'OperatorFlags',
    'Linear',
    'Square',
    'Symmetric',
    'Hermitian',
    'Idempotent',
    'Orthogonal',
    'Unitary',
    'Involutary',
    'ValidationError',
    'AdditionOperator',
    'CompositionOperator',
    'ScalarOperator',
]

verbose = True


class ValidationError(Exception):
    pass


class OperatorFlags(
    namedtuple(
        'OperatorFlags',
        [
            'LINEAR',
            'SQUARE',
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


def Linear(cls):
    cls.flags = cls.flags._replace(LINEAR=True)
    return cls


def Square(cls):
    cls.flags = cls.flags._replace(SQUARE=True)
    return cls


def Symmetric(cls):
    cls.flags = cls.flags._replace(LINEAR=True, SQUARE=True, SYMMETRIC=True)
    return cls


def Hermitian(cls):
    cls.flags = cls.flags._replace(LINEAR=True, SQUARE=True, REAL=False, HERMITIAN=True)
    return cls


def Idempotent(cls):
    cls.flags = cls.flags._replace(SQUARE=True, IDEMPOTENT=True)
    return cls


def Orthogonal(cls):
    cls.flags = cls.flags._replace(LINEAR=True, SQUARE=True, ORTHOGONAL=True)
    return cls


def Unitary(cls):
    cls.flags = cls.flags._replace(LINEAR=True, SQUARE=True, REAL=False, UNITARY=True)
    return cls


def Involutary(cls):
    cls.flags = cls.flags._replace(SQUARE=True, INVOLUTARY=True)
    return cls


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

        if dtype is not None:
            dtype = np.dtype(dtype)
        self.dtype = dtype

        if flags is not None:
            if isinstance(flags, tuple):
                self.flags = flags
            elif isinstance(flags, dict):
                self.flags = self.flags._replace(**flags)
            else:
                raise ValueError('Invalid input flags.')

        self._validate_flags()

        self.shapeout = shapeout if shapeout is not None else self.toshapeout(shapein)
        self.shapein = shapein if shapein is not None else self.toshapein(shapeout)

        if self.__class__ != 'Operator':
            self.__name__ = self.__class__.__name__
        elif direct is not None:
            self.__name__ = direct.__name__
            if self.__name__ == '<lambda>':
                self.__name__ = 'Operator'

    flags = OperatorFlags(*9 * (False,))
    shapein = None
    shapeout = None
    dtype = None

    direct = None
    transpose = None
    adjoint = None

    def conjugate_(self, input, output):
        self._direct(input.conjugate(), output)
        output[:] = output.conjugate()

    inverse = None
    inverse_transpose = None
    inverse_adjoint = None

    def inverse_conjugate(self, input, output):
        self.inverse(input.conjugate(), output)
        output[:] = output.conjugate()

    def __call__(self, input, output=None):
        if self.direct is None:
            raise NotImplementedError(
                'Call to ' + self.__name__ + ' is not im' 'plemented.'
            )
        input, output = self.validate_input(input, output)
        self.direct(input, output)
        return output

    @property
    def shape(self):
        shape = (np.product(self.shapeout), np.product(self.shapein))
        if shape[0] is None or shape[1] is None:
            return None
        return shape

    def reshapein(self, v):
        if self.shapein is None:
            raise ValueError(
                "The operator '" + self.__name__ + "' has an unde" "fined shapein."
            )
        return v.reshape(self.shapein)

    def reshapeout(self, v):
        if self.shapeout is None:
            raise ValueError(
                "The operator '" + self.__name__ + "' has an unde" "fined shapeout."
            )
        return v.reshape(self.shapeout)

    def toshapein(self, shape):
        if self.shapein is not None:
            return self.shapein
        return shape

    def toshapeout(self, shape):
        if self.shapeout is not None:
            return self.shapeout
        return shape

    def validate_input(self, input, output):
        """Returns input as ndarray and allocate output if necessary"""
        input = np.array(input, copy=False, subok=True, ndmin=1)
        if self.shapein is not None and self.shapein != input.shape:
            raise ValidationError(
                'The input of {0} has an incompatible shape '
                '{1}. Expected shape is {2}.'.format(
                    self.__name__, input.shapein, self.shapein
                )
            )
        shapeout = self.toshapeout(input.shape)
        output = self._allocate(
            shapeout, _get_dtypeout(input.dtype, self.dtype), output
        )
        return input, output

    @staticmethod
    def same_data(array1, array2):
        return (
            array1.__array_interface__['data'][0]
            == array2.__array_interface__['data'][0]
        )

    def todense(self, shapein=None):
        if not self.flags.LINEAR:
            raise TypeError('The operator is not linear.')
        shapein = shapein or self.shapein
        shapeout = self.toshapeout(shapein)
        m, n = np.product(shapeout), np.product(shapein)
        d = np.empty((n, m), self.dtype)
        v = np.zeros(n, self.dtype)
        for i in range(n):
            v[i] = 1
            self.direct(v.reshape(shapein), d[i, :].reshape(shapeout))
            v[i] = 0
        return d.T

    def matvec(self, v):
        v = self.reshapein(v)
        input, output = self.validate_input(v, None)
        self.direct(input, output)
        return output.ravel()

    def rmatvec(self, v):
        return self.T.matvec(v)

    def __str__(self):
        result = self.__name__
        if self.shapein is not None or self.shapeout is not None:
            result += ' [input:'
            if self.shapein is None:
                result += 'unconstrained'
            else:
                result += str(self.shapein).replace(' ', '')
            result += ', output:'
            if self.shapeout is None:
                result += 'unconstrained'
            else:
                result += str(self.shapeout).replace(' ', '')
            result += ']'
        return result

    def associated_operators(self):
        """
        By default, the operators returned by the C, T, H and I properties are
        instanciated from the methods provided in the operator's __init__.
        This method provides a way to override this behavior, by specifying the
        associated operators themselves as values in a dictionary, in which items are
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

        nbytes = dtype.itemsize * np.product(shape)
        if buf is not None:
            if buf.dtype != dtype:
                raise ValueError(
                    "Invalid output dtype '{0}'. Expected dtype i"
                    "s '{1}'.".format(buf.dtype, dtype)
                )
            if buf.nbytes != nbytes:
                raise ValueError(
                    'The output has invalid shape {0}. Expected s'
                    'hape is {1}.'.format(buf.shape, shape)
                )
            return buf.reshape(shape)

        if verbose:
            if nbytes < 1024:
                snbytes = str(nbytes) + ' bytes'
            else:
                snbytes = str(nbytes / 2**20) + ' MiB'
            print(
                'Info: Allocating '
                + str(shape).replace(' ', '')
                + ' '
                + dtype.type.__name__
                + ' = '
                + snbytes
                + ' in '
                + self.__name__
                + '.'
            )
        try:
            buf = np.empty(shape, dtype)
        except MemoryError:
            gc.collect()
            buf = np.empty(shape, dtype)
        return buf

    def _allocate_like(self, a, b):
        return self._allocate(a.shape, a.dtype, b)

    def _validate_flags(self):
        if self.dtype is None or self.dtype.kind != 'c':
            self.flags = self.flags._replace(REAL=True)

        if self.flags.REAL:
            if self.flags.SYMMETRIC:
                self.flags = self.flags._replace(HERMITIAN=True)
            if self.flags.HERMITIAN:
                self.flags = self.flags._replace(SYMMETRIC=True)
            if self.flags.ORTHOGONAL:
                self.flags = self.flags._replace(UNITARY=True)
            if self.flags.UNITARY:
                self.flags = self.flags._replace(ORTHOGONAL=True)

        if self.flags.ORTHOGONAL:
            if self.flags.IDEMPOTENT:
                self.flags = self.flags._replace(SYMMETRIC=True)
            if self.flags.SYMMETRIC:
                self.flags = self.flags._replace(IDEMPOTENT=True)

        if self.flags.UNITARY:
            if self.flags.IDEMPOTENT:
                self.flags = self.flags._replace(HERMITIAN=True)
            if self.flags.HERMITIAN:
                self.flags = self.flags._replace(IDEMPOTENT=True)

        if self.flags.INVOLUTARY:
            if self.flags.SYMMETRIC:
                self.flags = self.flags._replace(ORTHOGONAL=True)
            if self.flags.ORTHOGONAL:
                self.flags = self.flags._replace(SYMMETRIC=True)
            if self.flags.HERMITIAN:
                self.flags = self.flags._replace(UNITARY=True)
            if self.flags.UNITARY:
                self.flags = self.flags._replace(HERMITIAN=True)

        if self.flags.IDEMPOTENT:
            if any([self.flags.ORTHOGONAL, self.flags.UNITARY, self.flags.INVOLUTARY]):
                self.flags = self.flags._replace(
                    ORTHOGONAL=True, UNITARY=True, INVOLUTARY=True
                )

    def _generate_associated_operators(self):

        names = ('C', 'T', 'H', 'I', 'IC', 'IT', 'IH')
        ops = self.associated_operators()
        if not set(ops.keys()) <= set(names):
            raise ValueError(
                "Invalid associated operators. Expected operators are '{"
                "0}'".format(','.join(names))
            )

        if self.flags.REAL:
            C = self
        elif 'C' in ops:
            C = ops['C']
        else:
            C = Operator(self.conjugate_, dtype=self.dtype, flags=self.flags)
            C.__name__ += '.C'

        if 'T' in ops:
            T = ops['T']
        elif self.flags.SYMMETRIC:
            T = self
        else:
            T = Operator(self.transpose, dtype=self.dtype, flags=self.flags)
            T.__name__ += '.T'

        if 'H' in ops:
            H = ops['H']
        elif self.flags.HERMITIAN:
            H = self
        elif self.flags.REAL:
            H = T
        else:
            H = Operator(self.adjoint, dtype=self.dtype, flags=self.flags)
            H.__name__ += '.H'

        if 'I' in ops:
            I = ops['I']
        elif self.flags.ORTHOGONAL:
            I = T
        elif self.flags.UNITARY:
            I = H
        elif self.flags.INVOLUTARY:
            I = self
        else:
            I = Operator(self.inverse, dtype=self.dtype, flags=self.flags)
            I.__name__ += '.I'

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
            IC = Operator(self.inverse_conjugate, dtype=self.dtype, flags=self.flags)
            IC.__name__ += '.I.C'

        if 'IT' in ops:
            IT = ops['IT']
        elif self.flags.SYMMETRIC:
            IT = I
        elif self.flags.ORTHOGONAL:
            IT = self
        elif self.flags.UNITARY:
            IT = C
        elif self.flags.INVOLUTARY:
            IT = T
        else:
            IT = Operator(self.inverse_transpose, dtype=self.dtype, flags=self.flags)
            IT.__name__ += '.I.T'

        if 'IH' in ops:
            IH = ops['IH']
        elif self.flags.HERMITIAN:
            IH = I
        elif self.flags.ORTHOGONAL:
            IH = C
        elif self.flags.UNITARY:
            IH = self
        elif self.flags.INVOLUTARY:
            IH = H
        elif self.flags.REAL:
            IH = IT
        else:
            IH = Operator(self.inverse_adjoint, dtype=self.dtype, flags=self.flags)
            IH.__name__ += '.I.H'

        for op in (T, H, I, IC):
            op.shapein = self.shapeout
            op.shapeout = self.shapein
            op.reshapein = self.reshapeout
            op.reshapeout = self.reshapein
            op.toshapein = self.toshapeout
            op.toshapeout = self.toshapein

        for op in (C, IT, IH):
            op.shapein = self.shapein
            op.shapeout = self.shapeout

        # once all the associated operators are instanciated, we set all their
        # associated operators. To do so, we use the fact that the transpose, adjoint,
        # conjugate and inverse operators are commutative and involutary
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

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return self.matvec(other)
        return CompositionOperator([self, other])

    def __rmul__(self, other):
        if not _isscalar(other):
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


def asoperator(operator, shapein=None, shapeout=None):
    if isinstance(operator, Operator):
        if shapein and operator.shapein and shapein != operator.shapein:
            raise ValueError(
                'The input shapein ' + str(shapein) + ' is incom'
                'patible with that of the input ' + str(operator.shapein) + '.'
            )
        if shapeout and operator.shapeout and shapeout != operator.shapeout:
            raise ValueError(
                'The input shapeout ' + str(shapeout) + ' is inco'
                'mpatible with that of the input ' + str(operator.shapeout) + '.'
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
            output[:] = operator.matvec(input)

        def transpose(input, output):
            output[:] = operator.rmatvec(input)

        return Operator(
            direct=direct,
            transpose=transpose,
            shapein=shapein or operator.shape[1],
            shapeout=shapeout or operator.shape[0],
            dtype=operator.dtype,
        )

    if _isscalar(operator):
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
    def _validate_operands(cls, operands):
        operands = [asoperator(op) for op in operands]
        result = []
        for op in operands:
            if isinstance(op, cls):
                result.extend(op.operands)
            else:
                result.append(op)
        return result

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
            if isinstance(ops[i + 1], ScalarOperator):
                if isinstance(ops[i], ScalarOperator):
                    ops[i] = ScalarOperator(opn(ops[i].data, ops[i + 1].data))
                    del ops[i + 1]
                elif ops[i].flags.LINEAR:
                    ops[i], ops[i + 1] = ops[i + 1], ops[i]
                elif opn == np.multiply:
                    if ops[i + 1].data == 1:
                        del ops[i + 1]
            i -= 1
        if (
            len(ops) > 1
            and opn == np.multiply
            and isinstance(ops[0], ScalarOperator)
            and ops[0].data == 1
        ):
            del ops[0]

        return ops

    def __str__(self):
        result = Operator.__str__(self) + ':'
        components = []
        for operand in self.operands:
            components.extend(str(operand).split('\n'))
        result += '\n    ' + '\n    '.join(components)
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
            'LINEAR': all([op.flags.REAL for op in self.operands]),
            'REAL': all([op.flags.REAL for op in self.operands]),
            'SQUARE': self.shapein is not None
            and (self.shapein == self.shapeout)
            or all([op.flags.SQUARE for op in self.operands]),
        }
        CompositeOperator.__init__(self, flags=flags)
        self.work = [None, None]

    def associated_operators(self):
        return {
            'T': AdditionOperator([m.T for m in self.operands]),
            'H': AdditionOperator([m.H for m in self.operands]),
            'C': AdditionOperator([m.conjugate() for m in self.operands]),
        }

    def direct(self, input, output):
        operands = self.operands
        work = self.work

        # 1 operand: no temporaries
        if len(operands) == 1:
            operands[0].direct(input, output)
            return
        work[0] = self._allocate_like(output, work[0])

        # 2 operands: 1 temporary
        if len(operands) == 2:
            operands[0].direct(input, work[0])
            operands[1].direct(input, output)
            output += work[0]
            return

        # more than 2 operands, input == output: 2 temporaries
        if self.same_data(input, output):
            work[1] = self._allocate_like(output, work[1])
            operands[0].direct(input, work[0])
            for model in operands[1:-1]:
                model.direct(input, work[1])
                work[0] += work[1]
            operands[-1].direct(input, output)
            output += work[0]
            return

        # more than 2 operands, input != output: 1 temporary
        operands[0].direct(input, output)
        for model in self.operands[1:]:
            model.direct(input, work[0])
            output += work[0]

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
                raise ValidationError(
                    "Incompatible shape in operands: '"
                    + str(shapein)
                    + "' and '"
                    + str(shapein_)
                    + "'."
                )
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
                raise ValidationError(
                    "Incompatible shape in operands: '"
                    + str(shapeout)
                    + "' and '"
                    + str(shapeout_)
                    + "'."
                )
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
            'LINEAR': all([op.flags.REAL for op in self.operands]),
            'REAL': all([op.flags.REAL for op in self.operands]),
            'SQUARE': self.shapein is not None
            and (self.shapein == self.shapeout)
            or all([op.flags.SQUARE for op in self.operands]),
        }
        CompositeOperator.__init__(self, flags=flags)
        self.work = [None, None]

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

    def direct(self, input, output):
        operands = self.operands
        if len(operands) == 1:
            operands[0].direct(input, output)
            return

        # make the output variable available in the work pool
        self._set_output(output)

        i = input
        for model in reversed(self.operands):
            shapeout = model.toshapeout(input.shape)
            # get output from the work pool
            o = self._get_output(shapeout, input.dtype)
            model.direct(i, o)
            i = o

        # remove output from the work poll, to avoid side effects on the output
        self._del_output()

    @property
    def shapein(self):
        shape = None
        for model in self.operands:
            shape = model.toshapein(shape)
        return shape

    @shapein.setter
    def shapein(self, value):
        pass

    @property
    def shapeout(self):
        shape = None
        for model in reversed(self.operands):
            shape = model.toshapeout(shape)
        return shape

    @shapeout.setter
    def shapeout(self, value):
        pass

    def _get_output(self, shape, dtype):
        nbytes = np.product(shape) * dtype.itemsize
        if nbytes <= self.work[0].nbytes:
            buf = self.work[0]
        else:
            if self.work[1] is None or self.work[1].nbytes < nbytes:
                self.work[1] = self._allocate((nbytes,), np.int8)
            buf = self.work[1]

        return buf[:nbytes].view(dtype).reshape(shape)

    def _set_output(self, output):
        self.work[0] = output.ravel().view(np.int8)

    def _del_output(self):
        self.work[0] = None


@Symmetric
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
            flags = {'IDEMPOTENT': True, 'INVOLUTARY': True}
        else:
            flags = None

        Operator.__init__(
            self,
            lambda i, o: np.multiply(i, value, o),
            shapein=shapein,
            dtype=dtype,
            flags=flags,
        )

    def associated_operators(self):
        return {
            'C': ScalarOperator(np.conjugate(self.data)),
            'I': ScalarOperator(1 / self.data),
            'IC': ScalarOperator(np.conjugate(1 / self.data)),
        }

    def __str__(self):
        value = self.data.flat[0]
        if value == int(value):
            value = int(value)
        return str(value)


def _get_dtypeout(d1, d2):
    """Return dtype of greater type rank."""
    if d1 is None:
        return d2
    if d2 is None:
        return d1
    return np.find_common_type([d1, d2], [])


def _isscalar(data):
    """Hack around np.isscalar oddity"""
    return data.ndim == 0 if isinstance(data, np.ndarray) else np.isscalar(data)
