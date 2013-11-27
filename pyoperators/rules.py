from __future__ import division

import types
import core


class Rule(object):
    """
    Abstract class for operator rules.

    An operator rule is a relation that can be expressed by the sentence
    "'subjects' are 'predicate'". An instance of this class, when called with
    checks if the inputs are subjects to the rule, and returns the predicate
    if it is the case. Otherwise, it returns None.

    """

    def __init__(self, subjects, predicate):

        if not isinstance(subjects, (list, str, tuple)):
            raise TypeError("The input {0} is invalid.".format(subjects))

        subjects_ = self._split_subject(subjects)
        if any(
            not isinstance(s, str)
            and (not isinstance(s, type) or not issubclass(s, core.Operator))
            for s in subjects_
        ):
            raise TypeError("The subjects {0} are invalid.".format(subjects))
        if len(subjects_) == 0:
            raise ValueError('No rule subject is specified.')
        if len(subjects_) > 2:
            raise ValueError('No more than 2 subjects can be specified.')
        if not isinstance(self, UnaryRule) and len(subjects_) == 1:
            self.__class__ = UnaryRule
            self.__init__(subjects, predicate)
            return
        if not isinstance(self, BinaryRule) and len(subjects_) == 2:
            self.__class__ = BinaryRule
            self.__init__(subjects, predicate)
            return

        if '1' in subjects_:
            raise ValueError("'1' cannot be a subject.")
        if not isinstance(predicate, (str, types.FunctionType)):
            raise TypeError('Invalid predicate.')
        if isinstance(predicate, str) and '{' in predicate:
            raise ValueError("Predicate cannot be a subclass.")

        self.subjects = subjects_
        self.predicate = predicate

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return NotImplemented
        if self.subjects != other.subjects:
            return False
        if isinstance(self.predicate, types.FunctionType):
            if type(self.predicate) is not type(other.predicate):
                return False
            return self.predicate.func_code is other.predicate.func_code
        if isinstance(self.predicate, str):
            return self.predicate == other.predicate
        return self.predicate is other.predicate

    @staticmethod
    def _symbol2operator(op, symbol):
        if not isinstance(symbol, str):
            return symbol
        if symbol == '1':
            return core.IdentityOperator()
        if symbol == '.':
            return op
        try:
            return {'C': op._C, 'T': op._T, 'H': op._H, 'I': op._I}[symbol]
        except (KeyError):
            raise ValueError("Invalid symbol: '{0}'.".format(symbol))

    @classmethod
    def _split_subject(cls, subject):
        if isinstance(subject, str):
            subject = subject.split(',')
        if not isinstance(subject, (list, tuple)):
            raise TypeError('The rule subject is invalid.')
        subject = tuple(
            s.replace(' ', '') if isinstance(s, str) else s for s in subject
        )
        valid = '.,C,T,H,I,IC,IT,IH'.split(',')
        if any(
            (not isinstance(s, str) or s not in valid)
            and (not isinstance(s, type) or not issubclass(s, core.Operator))
            for s in subject
        ):
            raise ValueError('The rule subject is invalid.')
        return subject

    def __str__(self):
        subjects = [s if isinstance(s, str) else s.__name__ for s in self.subjects]
        return '{0} = {1}'.format(','.join(subjects), self.predicate)

    __repr__ = __str__


class UnaryRule(Rule):
    """
    Binary rule on operators.

    An operator unary rule is a relation that can be expressed by the sentence
    "'subject' is 'predicate'".

    Parameters
    ----------
    subject : str
        It defines the property of the operator for which the predicate holds:
            'C' : the operator conjugate
            'T' : the operator transpose
            'H' : the operator adjoint
            'I' : the operator adjoint
            'IC' : the operator inverse-conjugate
            'IT' : the operator inverse-transpose
            'IH' : the operator inverse-adjoint

    predicate : function or str
        What is returned by the rule when is applies. It can be:
            '1' : the identity operator
            '.' : the operator itself
            or a callable of one argument.

    Example
    -------
    >>> rule = UnaryRule('T', '.')
    >>> o = Operator()
    >>> oT = rule(o)
    >>> oT is o
    True

    """

    def __init__(self, subjects, predicate):
        super(UnaryRule, self).__init__(subjects, predicate)
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
        if not isinstance(predicate, core.Operator) and callable(predicate):
            predicate = predicate(reference)
        if not isinstance(predicate, core.Operator):
            raise TypeError('The predicate is not an operator.')
        return predicate


class BinaryRule(Rule):
    """
    Binary rule on operators.

    An operator rule is a relation that can be expressed by the sentence
    "'subjects' are 'predicate'". An instance of this class, when called with
    two input arguments checks if the inputs are subjects to the rule, and
    returns the predicate if it is the case. Otherwise, it returns None.

    Parameters
    ----------
    subjects : str
        It defines the relationship between the two subjects that must be
        verified for the rule to apply. It is a pair of two
        expressions. One has to be '.' and stands for the reference subject.
        It determines if the reference operator is on the right or left hand
        side of the operator pair. The other expression constrains the other
        subject, which must be:
            '.' : the reference operator itself.
            'C' : the conjugate of the reference object
            'T' : the transpose of the reference object
            'H' : the adjoint of the reference object
            or an Operator subclass.
        For instance, given a string 'C,.', the rule will apply to the inputs
        o1 and o2 if o1 is o2.C. For a condition ('.', DiagonalOperator), the
        rule will apply if o2 is a DiagonalOperator instance.

    predicate : function or str
        If the two objects o1, o2, are subjects of the rule, the predicate
        will be returned. The predicate can be '.', '1' or a callable
        of two arguments.

    Example
    -------
    >>> rule = BinaryRule('.,.', '.')
    >>> o = Operator()
    >>> rule(o, o) is o
    True
    >>> rule(o, IdentityOperator()) is None
    True

    """

    def __init__(self, subjects, predicate):
        super(BinaryRule, self).__init__(subjects, predicate)
        if len(self.subjects) != 2:
            raise ValueError('This is not a binary rule.')
        self.reference = 1 if self.subjects[1] == '.' else 0
        self.other = self.subjects[1 - self.reference]

    def __call__(self, o1, o2):

        reference, other = (o1, o2) if self.reference == 0 else (o2, o1)
        subother = self._symbol2operator(reference, self.other)

        if isinstance(subother, (type, tuple)):
            if subother is core.HomothetyOperator:
                subother = (core.HomothetyOperator, core.ZeroOperator)
            if not isinstance(other, subother):
                return None
        elif other != subother:
            return None

        predicate = self._symbol2operator(reference, self.predicate)
        if predicate is None:
            return None

        if not isinstance(predicate, core.Operator) and callable(predicate):
            predicate = predicate(o1, o2)
        if predicate is None:
            return None
        if isinstance(predicate, (list, tuple)) and len(predicate) == 1:
            predicate = predicate[0]
        if not isinstance(predicate, core.Operator) and not (
            isinstance(predicate, (list, tuple))
            and all(isinstance(o, core.Operator) for o in predicate)
        ):
            raise TypeError("The predicate '{0}' is not an operator.".format(predicate))
        return predicate
