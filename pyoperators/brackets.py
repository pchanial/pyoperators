from __future__ import division

"""
Module for operations on brackets.

A bracket is like Python's slice except that:
    - the stop boundary is inclusive
    - they cannot be empty (the start boundary is lesser than or equal to
      the stop one)
    - there is no step attribute.
Brackets are used by the CallingSequenceManager to determine where to split
a group of operators to resolve a requirement conflict.

"""


class bracket(object):
    """
    Implement [start:stop] brackets. Note that the type 'slice' cannot be
    subclassed.

    A bracket is like Python's slice except that:
        - the stop boundary is inclusive
        - they cannot be empty (the start boundary is lesser than or equal to
          the stop one)
        - there is no step attribute.

    """

    def __init__(self, start, stop):
        if stop < start:
            raise IndexError('Empty bracket.')
        self.start = start
        self.stop = stop

    def __cmp__(self, other):
        if self.start < other.start:
            return -1
        elif self.start == other.start:
            return self.stop.__cmp__(other.stop)
        return 1

    def __contains__(self, other):
        return self.start <= other.start and self.stop >= other.stop

    def __len__(self):
        return self.stop - self.start + 1

    def __str__(self):
        return '[{0}, {1}]'.format(self.start, self.stop)

    def __repr__(self):
        return 'bracket({0}, {1})'.format(self.start, self.stop)

    def copy(self):
        """
        Return a copy of the bracket.

        """
        return bracket(self.start, self.stop)

    def intersection_update(self, x):
        """
        Update a bracket with the intersection of itself and another.
        Raise an IndexError if there is no intersection.

        """
        if self.start > x.stop or x.start > self.stop:
            raise IndexError()
        self.start = max(self.start, x.start)
        self.stop = min(self.stop, x.stop)


class BracketList(list):
    """
    Hold a list of brackets.

    """

    def disjoint_intersections(self):
        """
        Return smallest disjoint brackets which contain at least one element
        in each bracket of the bracket list.

        This solution is not unique, for example the disjoint intersections of
        the brackets [1, 2], [2, 3], [3, 4] are:
            - [2, 2] and [3, 4]
            - [1, 2] and [3, 3]

        Currently, no more than two solutions are returned.

        """
        if len(self) <= 1:
            return (BracketList(self),)

        brackets = sorted(self)
        ascending = self._disjoint_intersection(brackets)
        descending = self._disjoint_intersection(brackets, reverse=True)
        if all(d in a for a, d in zip(ascending, descending)):
            return (ascending,)
        if all(a in d for a, d in zip(ascending, descending)):
            return (descending,)
        return (ascending, descending)

    def _disjoint_intersection(self, brackets, reverse=False):
        if reverse:
            brackets = reversed(brackets)
        brackets = iter(brackets)
        curr = next(brackets).copy()
        new_brackets = BracketList([curr])
        for b in brackets:
            try:
                curr.intersection_update(b)
            except IndexError:
                curr = b.copy()
                new_brackets.append(curr)
        if reverse:
            new_brackets.reverse()
        return new_brackets
