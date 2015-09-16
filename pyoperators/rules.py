from __future__ import absolute_import, division, print_function

import inspect
import types
import os
from . import config
from .core import HomothetyOperator, IdentityOperator, Operator, ZeroOperator
from .warnings import warn, PyOperatorsWarning
import collections

__all__ = ['rule_manager']
_triggers = {}
_default_triggers = {
    'inplace': False,
    'none': False}
_description_triggers = {
    'inplace': 'Allow inplace simplifications',
    'none': 'Inhibit all rule simplifications'}


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
        if any(not isinstance(s, str) and (not isinstance(s, type) or
               not issubclass(s, Operator)) for s in subjects_):
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
            return self.predicate.__code__ is other.predicate.__code__
        if isinstance(self.predicate, str):
            return self.predicate == other.predicate
        return self.predicate is other.predicate

    @staticmethod
    def _symbol2operator(op, symbol):
        if not isinstance(symbol, str):
            return symbol
        if symbol == '1':
            return IdentityOperator()
        if symbol == '.':
            return op
        try:
            return {'C': op._C,
                    'T': op._T,
                    'H': op._H,
                    'I': op._I}[symbol]
        except (KeyError):
            raise ValueError("Invalid symbol: '{0}'.".format(symbol))

    @classmethod
    def _split_subject(cls, subject):
        if isinstance(subject, str):
            subject = subject.split(',')
        if not isinstance(subject, (list, tuple)):
            raise TypeError('The rule subject is invalid.')
        subject = tuple(s.replace(' ', '') if isinstance(s, str) else s
                        for s in subject)
        valid = '.,C,T,H,I,IC,IT,IH'.split(',')
        if any((not isinstance(s, str) or s not in valid) and
               (not isinstance(s, type) or not issubclass(s, Operator))
               for s in subject):
            raise ValueError('The rule subject is invalid.')
        return subject

    def __str__(self):
        subjects = [s if isinstance(s, str) else s.__name__
                    for s in self.subjects]
        spredicate = ' '.join(s.strip() for s in inspect.getsource(
            self.predicate).split('\n')) \
            if isinstance(self.predicate, types.LambdaType) \
            else self.predicate
        return '{0} = {1}'.format(','.join(subjects), spredicate)

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
        if isinstance(predicate, collections.Callable) or predicate in ('.', '1'):
            return
        raise ValueError("Invalid predicate: '{0}'.".format(predicate))

    def __call__(self, reference):
        predicate = self._symbol2operator(reference, self.predicate)
        if predicate is None:
            return None
        if not isinstance(predicate, Operator) and isinstance(predicate, collections.Callable):
            predicate = predicate(reference)
        if not isinstance(predicate, Operator):
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
        self.other = self.subjects[1-self.reference]

    def __call__(self, o1, o2):

        reference, other = (o1, o2) if self.reference == 0 else (o2, o1)
        subother = self._symbol2operator(reference, self.other)

        if isinstance(subother, (type, tuple)):
            if subother is HomothetyOperator:
                subother = (HomothetyOperator, ZeroOperator)
            if not isinstance(other, subother):
                return None
        elif other != subother:
            return None

        predicate = self._symbol2operator(reference, self.predicate)
        if predicate is None:
            return None

        if not isinstance(predicate, Operator) and isinstance(predicate, collections.Callable):
            predicate = predicate(o1, o2)
        if predicate is None:
            return None
        if isinstance(predicate, (list, tuple)) and len(predicate) == 1:
            predicate = predicate[0]
        if not isinstance(predicate, Operator) \
           and not (isinstance(predicate, (list, tuple))
                    and all(isinstance(o, Operator)
                            for o in predicate)):
            raise TypeError("The predicate '{0}' is not an operator.".format(
                            predicate))
        return predicate


class RuleManager(object):
    """
    Manage a set of rule prescriptions.

    It is a proxy for the global dictionary that contains the rule names
    and values. It also provides a context manager to change the rules inside
    a with statement.
    Rule defaults can be stored in a file 'rules.txt' in the user directory
    pyoperators.config.LOCAL_PATH.

    Examples
    --------
    To prevent rule simplifications:
    >>> from pyoperators.rules import rule_manager
    >>> rule_manager['none'] = True

    To re-enable rule simplifications:
    >>> rule_manager['none'] = False

    or:
    >>> with rule_manager(none=True):
    ...     # in this context, operator simplification rules are inhibited
    ...     print(rule_manager['none'])
    True
    >>> print(rule_manager['none'])
    False

    It is possible to nest contexts:
    >>> print(rule_manager['none'])
    False
    >>> with rule_manager(none=True) as new_rule_manager:
    ...     print(rule_manager['none'])
    ...     with new_rule_manager(none=False):
    ...         print(rule_manager['none'])
    ...     print(rule_manager['none'])
    True
    False
    True
    >>> print(rule_manager['none'])
    False

    """
    def __init__(self):
        self._deferred_triggers = {}
        if len(self) == 0:
            self.update(_default_triggers)
            self._update_user_default_triggers()

    def __call__(self, **keywords):
        for key in keywords:
            if key not in self:
                raise KeyError('Unknown rule: {!r}'.format(key))
        self._deferred_triggers = keywords
        return self

    def __enter__(self):
        self._old_triggers = self.copy()
        self.update(self._deferred_triggers)
        return RuleManager()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _triggers
        _triggers = self._old_triggers
        return False

    def __getitem__(self, key):
        return _triggers[key]

    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError('Unknown rule: {!r}'.format(key))
        _triggers[key] = value

    def __contains__(self, key):
        return key in _triggers

    def __iter__(self):
        return iter(sorted(_triggers.keys()))

    def __len__(self):
        return len(_triggers)

    def __str__(self):
        nk = max(len(k) for k in self)
        nv = max(len(repr(v)) for v in self.values())
        s = '{0:' + str(nk) + '} = {1!r:' + str(nv) + '}  # {2}'
        return '\n'.join(s.format(k, self[k], _description_triggers.get(k, ''))
                         for k in self)

    def clear(self):
        """ Clear the global rule dictionary. """
        _triggers.clear()

    def copy(self):
        """ Copy the global rule dictionary. """
        return _triggers.copy()

    def get(self, k, *args):
        """ Get a rule value in the global rule dictionary. """
        return _triggers.get(k, *args)

    def items(self):
        """ Return the global rule items. """
        return _triggers.items()

    def keys(self):
        """ Return the global rule names. """
        return _triggers.keys()

    def pop(self, k, *args):
        """ Pop a given item from the global rule dictionary. """
        return _triggers.pop(k, *args)

    def popitem(self):
        """ Pop any item from the global rule dictionary. """
        return _triggers.popitem()

    def register(self, rule, default, description):
        """ Add a new rule. """
        # should not be called in a managed context
        if not isinstance(rule, str):
            raise TypeError('The rule is not a string.')
        if not isinstance(description, str):
            raise TypeError('The rule description is not a string.')
        rule = rule.lower()
        _triggers[rule] = default
        _description_triggers[rule] = description

    def update(self, *args, **keywords):
        """ Update the global rule dictionary. """
        _triggers.update(*args, **keywords)

    def values(self):
        """ Return the global rule values. """
        return _triggers.values()

    def _update_user_default_triggers(self):
        # read user 'rules.txt' to update defaults
        path = os.path.join(config.LOCAL_PATH, 'rules.txt')
        if not os.path.exists(path):
            return
        if not os.access(path, os.R_OK):
            warn('The file {0!r} cannot be read.'.format(path),
                 PyOperatorsWarning)
            return
        with open(path) as f:
            for iline, line in enumerate(f.readlines()):
                line = line.strip()
                line_orig = line
                try:
                    index = line.index('#')
                except ValueError:
                    pass
                else:
                    line = line[:index].rstrip()
                try:
                    index = line.index('=')
                except ValueError:
                    if len(line) == 0:
                        continue
                    warn('In file {0!r}, line {1} does not define a rule: {2!r'
                         '}.'.format(path, iline + 1, line_orig),
                         PyOperatorsWarning)
                    continue
                key = line[:index].rstrip().lower()
                value = line[index+1:].lstrip()
                try:
                    value = eval(value, {})
                except Exception:
                    warn('In file {0!r}, line {1}: {2!r} cannot be evaluated'.
                         format(path, iline+1, value), PyOperatorsWarning)
                    continue
                _triggers[key] = value

    __repr__ = __str__

rule_manager = RuleManager()
