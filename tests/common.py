from pyoperators import Operator, decorators

@decorators.real
@decorators.symmetric
class HomothetyOutplaceOperator(Operator):
    def __init__(self, value, **keywords):
        Operator.__init__(self, **keywords)
        self.value = value
    def direct(self, input, output):
        output[...] = self.value * input


class Stretch(Operator):
    """ Stretch input array by replicating it by a factor of 2. """
    def __init__(self, axis, **keywords):
        self.axis = axis
        if self.axis < 0:
            self.slice = [Ellipsis] + (-self.axis) * [slice(None)]
        else:
            self.slice = (self.axis+1) * [slice(None)] + [Ellipsis]
        Operator.__init__(self, **keywords)
    def direct(self, input, output):
        self.slice[self.axis] = slice(0,None,2)
        output[self.slice] = input
        self.slice[self.axis] = slice(1,None,2)
        output[self.slice] = input
    def reshapein(self, shape):
        shape_ = list(shape)
        shape_[self.axis] *= 2
        return shape_
    def reshapeout(self, shape):
        shape_ = list(shape)
        shape_[self.axis] //= 2
        return shape_
