"""
MPI-wrapper module for non-MPI enabled platforms.
"""
_g = globals()
_constants = 'SUM', 'MIN', 'MAX', 'BYTE', 'FLOAT', 'DOUBLE', 'IN_PLACE'
for i, c in enumerate(_constants):
    _g[c] = i

class Comm(object):
    def __init__(self, rank, size):
        self.rank = rank
        self.size = size
    def Get_rank(self):
        return self.rank
    def Get_size(self):
        return self.size
    def allgather(self,sendobj=None, recvobj=None):
        return [sendobj]
    def allreduce(self, sendobj=None, recvobj=None, op=SUM):
        return sendobj
    def Allgatherv(self, i, o, op=None):
        if i == IN_PLACE:
            return
        o[0][...] = i[0]
    def Allreduce(self, i, o, op=None):
        if i == IN_PLACE:
            return
        o[0][...] = i[0]
    def Barrier(self):
        pass
    @staticmethod
    def f2py(fcomm):
         return COMM_SELF
    def py2f(self):
        return 0

def Get_processor_name():
    return ''

COMM_NULL = Comm(0, 0)
COMM_SELF = Comm(0, 1)
COMM_WORLD = COMM_SELF

del _g, _constants
