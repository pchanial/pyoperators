"""
MPI-wrapper module for non-MPI enabled platforms.
"""

import builtins as _builtins
from itertools import count as _count

_inc = _count()

SUM = next(_inc)
MIN = next(_inc)
MAX = next(_inc)
PROD = next(_inc)
BAND = next(_inc)
BOR = next(_inc)
BXOR = next(_inc)
LAND = next(_inc)
LOR = next(_inc)
LXOR = next(_inc)
MAXLOC = next(_inc)
MINLOC = next(_inc)
BOOL = next(_inc)
BYTE = next(_inc)
C_BOOL = next(_inc)
C_COMPLEX = next(_inc)
C_DOUBLE_COMPLEX = next(_inc)
C_FLOAT_COMPLEX = next(_inc)
C_LONG_DOUBLE_COMPLEX = next(_inc)
CHAR = next(_inc)
CHARACTER = next(_inc)
WCHAR = next(_inc)
COMPLEX = next(_inc)
COMPLEX4 = next(_inc)
COMPLEX8 = next(_inc)
COMPLEX16 = next(_inc)
COMPLEX32 = next(_inc)
DOUBLE = next(_inc)
DOUBLE_COMPLEX = next(_inc)
DOUBLE_INT = next(_inc)
DOUBLE_PRECISION = next(_inc)
F_BOOL = next(_inc)
F_COMPLEX = next(_inc)
F_DOUBLE = next(_inc)
F_DOUBLE_COMPLEX = next(_inc)
F_FLOAT = next(_inc)
F_FLOAT_COMPLEX = next(_inc)
F_INT = next(_inc)
FLOAT = next(_inc)
FLOAT_INT = next(_inc)
INT = next(_inc)
INT8_T = next(_inc)
INT16_T = next(_inc)
INT32_T = next(_inc)
INT64_T = next(_inc)
INT_INT = next(_inc)
INTEGER = next(_inc)
INTEGER1 = next(_inc)
INTEGER2 = next(_inc)
INTEGER4 = next(_inc)
INTEGER8 = next(_inc)
INTEGER16 = next(_inc)
LOGICAL = next(_inc)
LOGICAL1 = next(_inc)
LOGICAL2 = next(_inc)
LOGICAL4 = next(_inc)
LOGICAL8 = next(_inc)
LONG = next(_inc)
LONG_DOUBLE = next(_inc)
LONG_DOUBLE_INT = next(_inc)
LONG_INT = next(_inc)
LONG_LONG = next(_inc)
PACKED = next(_inc)
REAL = next(_inc)
REAL2 = next(_inc)
REAL4 = next(_inc)
REAL8 = next(_inc)
REAL16 = next(_inc)
SHORT = next(_inc)
SHORT_INT = next(_inc)
SIGNED_CHAR = next(_inc)
SIGNED_INT = next(_inc)
SIGNED_LONG = next(_inc)
SIGNED_LONG_LONG = next(_inc)
SIGNED_SHORT = next(_inc)
SINT8_T = next(_inc)
SINT16_T = next(_inc)
SINT32_T = next(_inc)
SINT64_T = next(_inc)
TWOINT = next(_inc)
UINT8_T = next(_inc)
UINT16_T = next(_inc)
UINT32_T = next(_inc)
UINT64_T = next(_inc)
UNSIGNED = next(_inc)
UNSIGNED_CHAR = next(_inc)
UNSIGNED_INT = next(_inc)
UNSIGNED_LONG = next(_inc)
UNSIGNED_LONG_LONG = next(_inc)
UNSIGNED_SHORT = next(_inc)
WCHAR = next(_inc)
IN_PLACE = next(_inc)
KEYVAL_INVALID = next(_inc)


class Comm:
    _keyvals = {}  # class attribute

    def __init__(self, rank, size):
        self.rank = rank
        self.size = size
        self._attr = {}

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def allgather(self, sendobj=None, recvobj=None):
        return [sendobj]

    def allreduce(self, sendobj=None, recvobj=None, op=SUM):
        return sendobj

    def bcast(self, obj=None, root=0):
        return obj

    def gather(self, sendobj=None, recvobj=None, root=0):
        return [sendobj]

    def Allgatherv(self, i, o, op=None):
        if isinstance(i, int) and i == IN_PLACE:
            return
        if isinstance(i, (list, tuple)):
            i = i[0]
        o[0][...] = i

    def Allreduce(self, i, o, op=None):
        if isinstance(i, int) and i == IN_PLACE:
            return
        if isinstance(i, (list, tuple)):
            i = i[0]
        o[0][...] = i

    def Barrier(self):
        pass

    def Dup(self):
        return Comm(self.rank, self.size)

    Clone = Dup

    def Free(self):
        return

    def Split(self, color=0, key=0):
        return Comm(self.rank, self.size)

    @classmethod
    def Create_keyval(cls, copy_fn=None, delete_fn=None):
        if len(cls._keyvals) == 0:
            id = 1
        else:
            id = max(cls._keyvals.keys()) + 1
        cls._keyvals[id] = (copy_fn, delete_fn)
        return id

    @classmethod
    def Free_keyval(cls, keyval):
        if keyval not in cls._keyvals:
            raise ValueError('Invalid keyval.')
        del cls._keyvals[keyval]

    def Delete_attr(self, keyval):
        if keyval not in self._attr:
            raise ValueError('Invalid keyval.')
        del self._attr[keyval]

    def Get_attr(self, keyval):
        if keyval not in self._keyvals:
            raise ValueError('Invalid keyval.')
        if keyval not in self._attr:
            return None
        return self._attr[keyval]

    def Set_attr(self, keyval, attrval):
        if keyval not in self._keyvals:
            raise ValueError('Invalid keyval.')
        self._attr[keyval] = attrval

    def Create_cart(self, dims, periods=None, reorder=False):
        return Cartcomm(self.rank, self.size)

    @staticmethod
    def f2py(fcomm):
        return COMM_SELF

    def py2f(self):
        return 0


class Cartcomm(Comm):
    def Sub(self, remain_dims):
        return Comm(self.rank, self.size)


def Get_processor_name():
    import platform

    return platform.node()


COMM_NULL = Comm(0, 0)
COMM_SELF = Comm(0, 1)
COMM_WORLD = Comm(0, 1)


class Exception(_builtins.Exception):
    """Exception.__init__(self, int ierr=0)"""

    def __init__(self, ierr=0):
        pass
