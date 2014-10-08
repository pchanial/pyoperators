"""
MPI-wrapper module for non-MPI enabled platforms.
"""
try:
    import builtins
except ImportError:
    import __builtin__ as builtins

_g = globals()
_constants = ('SUM', 'MIN', 'MAX', 'PROD', 'BAND', 'BOR', 'BXOR',
              'LAND', 'LOR', 'LXOR', 'MAXLOC', 'MINLOC',
              'BOOL', 'BYTE',
              'C_BOOL', 'C_COMPLEX', 'C_DOUBLE_COMPLEX', 'C_FLOAT_COMPLEX',
              'C_LONG_DOUBLE_COMPLEX',
              'CHAR', 'CHARACTER', 'WCHAR',
              'COMPLEX', 'COMPLEX4', 'COMPLEX8', 'COMPLEX16', 'COMPLEX32',
              'DOUBLE', 'DOUBLE_COMPLEX', 'DOUBLE_INT', 'DOUBLE_PRECISION',
              'F_BOOL', 'F_COMPLEX', 'F_DOUBLE', 'F_DOUBLE_COMPLEX', 'F_FLOAT',
              'F_FLOAT_COMPLEX', 'F_INT',
              'FLOAT', 'FLOAT_INT',
              'INT', 'INT8_T', 'INT16_T', 'INT32_T', 'INT64_T', 'INT_INT',
              'INTEGER', 'INTEGER1', 'INTEGER2', 'INTEGER4', 'INTEGER8',
              'INTEGER16',
              'LOGICAL', 'LOGICAL1', 'LOGICAL2', 'LOGICAL4', 'LOGICAL8',
              'LONG', 'LONG_DOUBLE', 'LONG_DOUBLE_INT', 'LONG_INT', 'LONG_LONG',
              'PACKED',
              'REAL', 'REAL2', 'REAL4', 'REAL8', 'REAL16',
              'SHORT', 'SHORT_INT',
              'SIGNED_CHAR', 'SIGNED_INT', 'SIGNED_LONG', 'SIGNED_LONG_LONG',
              'SIGNED_SHORT',
              'SINT8_T', 'SINT16_T', 'SINT32_T', 'SINT64_T',
              'TWOINT',
              'UINT8_T', 'UINT16_T', 'UINT32_T', 'UINT64_T',
              'UNSIGNED', 'UNSIGNED_CHAR', 'UNSIGNED_INT', 'UNSIGNED_LONG',
              'UNSIGNED_LONG_LONG', 'UNSIGNED_SHORT',
              'WCHAR',
              'IN_PLACE', 'KEYVAL_INVALID')
for i, c in enumerate(_constants):
    _g[c] = i


class Comm(object):
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


class Exception(builtins.Exception):
    """ Exception.__init__(self, int ierr=0) """
    def __init__(self, ierr=0):
        pass

del _g, _constants, builtins
