"""
This module handles the chaining of operations that need to be performed
to call an operator or a set of operators given a calling sequence.

"""
from __future__ import division

import copy
import numpy as np
from .brackets import BracketList
from .memory import MEMORY_ALIGNMENT, empty, isvalid, view
from .utils import (
    groupbykey,
    ifirst,
    isalias,
    operation_assignment,
    product,
    tointtuple,
)
import core


class Requirement(object):
    """
    Buffer requirements.

    The composition op2 * op1 uses the following 3 requirements:
        - op1's input
        - op1's output (same as op2's input)
        - op2's output.

    """

    def __init__(self):
        self.dtype = None
        self.shape = None
        self.strides = None
        self.aligned = True  # XXX should be None

    @property
    def nbytes(self):
        return product(self.shape) * self.dtype.itemsize

    def copy(self):
        return copy.copy(self)

    def __str__(self):
        return '({0}, {1}, {2}, {3})'.format(
            self.dtype, self.shape, self.strides, self.aligned
        )

    __repr__ = __str__


class CallingSequence(object):
    """
    This class computes the required operations to process the composition
    of a chain of operators according to their requirements (such as alignment
    or contiguity) or capabilities (in-place or out-of-place) for a given
    calling sequence.

    """

    def __init__(
        self,
        operators,
        (dtypein, shapein, stridesin, alignedin),
        (dtypeout, shapeout, stridesout, alignedout),
        preserve_input,
        preserve_output,
        alias,
    ):
        self.operators = list(operators)
        self.requirements = [Requirement() for _ in range(len(operators) + 1)]
        self.groups = self.compute_groups()
        self.analyse_preserve_input(preserve_input)
        self.analyse_preserve_output(preserve_output)
        self.analyse_dtype(dtypein, dtypeout)
        self.analyse_shape(shapein, shapeout)
        self.analyse_requirements(alignedin, stridesin, alignedout, stridesout)
        self.analyse_alias(alias)
        self.analyse_buffer_usage()

    def __call__(self, input, output, operation):
        operators = self.operators.__iter__()
        requirements = self.requirements[1:].__iter__()
        noperators = len(self.operators)
        output_in_pool = False
        bufferin = input
        input_ = input

        # loop over the groups
        iop = 0
        for igroup, (n, size, use_output) in enumerate(
            zip(self.groups, self.nbytes, self.use_output)
        ):

            if igroup > 0 or not self.first_group_is_inplace:

                # get the buffer of the group
                if output_in_pool:
                    core._pool.remove(output)
                    output_in_pool = False
                if use_output:
                    bufferout = output
                else:
                    bufferout = core._pool.extract(size, np.int8)
                    if self.reuse_output and bufferin is not output:
                        # make buffer 'output' available for the temporaries
                        # of the group operators
                        core._pool.add(output)
                        output_in_pool = True

                # apply the out-of-place operator
                op = next(operators)
                r = next(requirements)
                output_ = view(bufferout, r.shape, r.dtype, r.strides)
                if iop < noperators - 1 or not self.preserve_output:
                    op.direct(input_, output_)
                else:
                    op.direct(input_, output_, operation=operation)
                input_ = output_
                iop += 1
                n -= 1

                # put the input buffer back in the pool, if necessary
                if bufferin is not output:
                    if bufferin is not input or self.reuse_input:
                        core._pool.add(bufferin)

            else:
                bufferout = bufferin

            # loop over the in-place operators of the group
            for iop in range(iop, iop + n):
                op = next(operators)
                r = next(requirements)
                output_ = view(bufferout, r.shape, r.dtype, r.strides)
                op.direct(input_, output_)
                input_ = output_

            bufferin = bufferout

        if output_in_pool:
            core._pool.remove(output)

        if self.reuse_input:
            core._pool.remove(input)

        return output

    def compute_groups(self):
        """
        Partition the operator chain into groups that will work on the same
        memory buffer.

        """
        outplace = [not o.flags.inplace for o in self.operators]
        keys = np.cumsum(outplace)
        groups = []
        for junk, ops in groupbykey(self.operators, keys):
            groups.append(len(ops))
        return groups

    def compute_nbytes(self):
        """
        Compute the minimum size of the buffers that will used by the groups.

        """
        start = 0
        nbytes = []
        for group in self.groups:
            size = max(b.nbytes for b in self.requirements[start : start + group])
            nbytes.append(size)
            start += group
        return nbytes

    def analyse_preserve_input(self, preserve_input):
        """Ensure that the input buffer is preserved."""
        self.preserve_input = preserve_input
        self.first_group_is_inplace = (
            not preserve_input and self.operators[0].flags.inplace
        )
        if not preserve_input:
            return
        op = self.operators[0]
        if not op.flags.destroy_input and op.flags.outplace:
            return
        self.operators.insert(0, core._C)
        self.requirements.insert(0, Requirement())
        if op.flags.inplace:
            self.groups[0] += 1
        else:
            self.groups.insert(0, 1)

    def analyse_preserve_output(self, preserve_output):
        """Ensure that the output buffer is preserved."""
        self.preserve_output = preserve_output
        if not preserve_output:
            return
        if self.operators[-1].flags.inplace_reduction:
            # XXX no warranty the particular operation will be handled...
            return
        self.operators.append(core._C)
        self.requirements.append(Requirement())
        self.groups.append(1)

    def analyse_dtype(self, dtypein, dtypeout):
        dtype = dtypein
        self.requirements[0].dtype = dtype
        for op, req in zip(self.operators, self.requirements[1:]):
            req.dtype = dtype = np.dtype(op.redtypein(dtype))
        if dtypeout is not None and dtype != dtypeout:
            raise TypeError(
                "The output dtype '{0}' is not the expected one '{1}'.".format(
                    dtypeout, dtype
                )
            )

    def analyse_shape(self, shapein, shapeout):
        shape = shapein
        self.requirements[0].shape = shape
        for op, req in zip(self.operators, self.requirements[1:]):
            req.shape = shape = tointtuple(op.reshapein(shape))
        if shapeout is not None and shape is not None and shape != shapeout:
            raise TypeError(
                "The output shape '{0}' is not the expected one '{1}'.".format(
                    shapeout, shape
                )
            )

        if shape is None:
            shape = shapeout
        self.requirements[-1].shape = shape
        for op, req in zip(self.operators[::-1], self.requirements[-2::-1]):
            shape = tointtuple(op.reshapeout(shape))
            if req.shape is None:
                req.shape = shape
            elif shape is None:
                shape = req.shape
            elif shape != req.shape:
                raise ValueError('Incompatible shape in composition.')

    def analyse_requirements(self, alignedin, stridesin, alignedout, stridesout):
        """
        Ensure that the operators can operate on the buffer for their group.

        """
        bl = BracketList()
        # self.find_candidates_alignment(bl, alignedin, alignedout)
        # self.find_candidates_strides(bl, stridesin, stridesout)
        # self.find_candidates_nbytes(bl)
        # self.choose_candidates(bl)

    def find_candidates_alignment(self, bl, alignedin, alignedout):
        self.requirements[0].aligned = alignedin
        if not alignedin:
            ops = self.operators[self.get_group_slice(0)]
            try:
                index = ifirst(ops, lambda x: x.flags.aligned_input)
                self.split_group_before(0, index)
            except ValueError:
                pass

        self.requirements[-1].aligned = alignedout
        if not alignedout:
            ops = self.operators[self.get_group_slice(-1)]
            try:
                index = ifirst(ops, lambda x: x.flags.aligned_output, reverse=True)
                self.split_group_after(-1, index)
            except ValueError:
                pass

    def find_candidates_strides(self, bl, stridesin, stridesout):
        # stage 1: find operators which impose strides on the buffers
        forward = len(self.requirements) * [None]
        forward[0] = self.requirements[0].strides
        for iop, (op, b) in enumerate(zip(self.operators, self.requirements[:-1])):
            strides = op.restridesin(None, b.shape, b.dtype)
            forward[iop] = strides

        backward = len(self.requirements) * [None]
        backward[-1] = self.requirements[-1].strides
        for iop, (op, b) in renumerate(zip(self.operators, self.requirements[1:])):
            strides = op.restridesout(None, b.shape, b.dtype)
            backward[iop] = strides

        for op, bin, bout in self._iter_operators():
            bout.strides = op.restridesin(bin.strides, bin.shape, bin.dtype)

        # stage 2: solve conflicts by looping over them
        conflicts = self._iter_conflict(forward, backward)
        for c, n in conflicts:
            iop = c.start
            strides = forward[c.start]
            strides_array = [s]
            for op, b in zip(self.operators[c], self.requirements[c]):
                strides = self.restridesin(s, b.shape, b.dtype)
                strides_array.append(s)

            if strides == backward[c.stop - 1]:
                # not a real conflict
                for b, s in zip(self.requirements[c.start : c.stop + 1], strides_array):
                    b.strides = s
            else:
                self.conflicts += (c, {'strides': [forward[c.start], backward[c.stop]]})

        nonconflicts = self._iter_nonconflict(forward, backward)
        for nc in nonconflicts:
            self.requirements[nc.start].strides = backward[nc.start]
            for op, bin, bout in self._iter_operators(nc):
                bout.strides = self.restridesin(bin.strides, bin.shape, bin.dtype)

    def _iter_conflicts(self, forward, backward):
        """
        Iterate through the slices that bracket forward and backward values.

        """
        start = 0
        flag = True
        for i, (f, b) in enumerate(zip(forward, backward)):
            if f is not None:
                start = i
                flag = True
            if b is not None:
                if flag:
                    yield slice(start, i + 1)
                    start = i
                    flag = False

    def _iter_nonconflicts(self, forward, backward):
        """
        Iterate through the slices in-between brackets of forward and backward
        values.

        """
        start = 0
        flag = True
        for i, (f, b) in enumerate(zip(forward, backward)):
            if f is not None:
                if flag:
                    start = i
                    flag = True
            if b is not None:
                if flag:
                    start = i
                    flag = False

    def find_candidates_strides(self, bl, stridesin, stridesout):
        strides = stridesin
        self.requirements[0].strides = strides
        shape = self.requirements[0].shape
        dtype = self.requirements[0].dtype
        iop = 0
        for igroup, (ops, bufs) in enumerate(self.iter_groups()):
            for op, buf in zip(ops, bufs):
                inplace = iop > 0 or igroup == 0 and self.first_group_is_inplace
                strides = op.restridesin(strides, shape, dtype, inplace)
                buf.strides = strides
                shape = buf.shape
                dtype = buf.dtype
                iop += 1

        if stridesout is not None and strides is not None and strides != stridesout:
            pass

        igroup = len(self.groups)
        for ops, bufs in self.iter_groups(reverse=True):
            igroup -= 1
            iop = len(ops) - 1
            for op, buf in zip(ops, bufs):
                inplace = iop > 0 or igroup == 0 and self.first_group_is_inplace
                strides = op.restridesout(strides, shape, dtype, inplace)
                buf.strides = strides
                shape = buf.shape
                dtype = buf.dtype
                iop -= 1

    def find_candidates_nbytes(self, bl):
        """
        Make sure that the output buffer is large enough to be used for
        the last group.

        """
        n = self.groups[-1]
        if n == 1:
            return
        nbytes = self.requirements[-1].nbytes
        try:
            index = ifirst(self.requirements[-n:-1], lambda x: x.nbytes > nbytes)
            self.split_group_before(-1, index + 1)
        except ValueError:
            pass

    def choose_candidates(self, bl):
        """
        Once the candidates for the new buffers are found, let's find
        the candidates that
            - minimise the number of groups (hence number of out-of-place
        operations)
            - minimise the number of operations on the input and output
        buffers if they are not contiguous or not aligned.
            - and that would make the out-of-place operation on the smallest
        possible buffer

        """
        for bracket in reversed(bl.disjoint_intersections()):
            try:
                self.convert2outplace(bracket)
            except IndexError:
                self.insert_copy(bracket)

    def analyse_alias(self, alias):
        self.alias = alias
        if not alias:
            return

    def analyse_buffer_usage(self):
        self.nbytes = self.compute_nbytes()
        noutplaces = len(self.groups) - self.first_group_is_inplace
        b = self.requirements[0]
        self.reuse_input = (
            not self.preserve_input
            and noutplaces > 2
            and isvalid(b.shape, b.dtype, b.strides)
        )
        b = self.requirements[-1]
        self.reuse_output = not self.preserve_output and isvalid(
            b.shape, b.dtype, b.strides
        )
        if self.reuse_input and self.reuse_output and self.alias:
            self.reuse_input = False

        self.use_output = (len(self.groups) - 1) * [False] + [True]
        if not self.reuse_output:
            return

        # find which groups could use the output buffer
        igroup = len(self.groups) - 3
        nbytes = self.nbytes[-1]
        while igroup >= 0:
            if self.nbytes[igroup] <= nbytes:
                self.use_output[igroup] = True
                igroup -= 2
            else:
                igroup -= 1

    def split_group(self, igroup, index):
        raise
        group = self.groups[igroup]
        buffers = group.buffers
        n = len(buffers)
        if index < 0 or index > n:
            raise ValueError(
                "Invalid index value '{0}' ({1}<=index<={2}).".format(index, 0, n)
            )
        if index in (0, n):
            group.insert_copy(index)
            return

    def insert_copy(self, bracket):
        """
        Insert a CopyOperator at of the ends of a bracket. The end which
        operates on the smallest buffer is choosen.

        """
        start = bracket.start
        stop = bracket.stop
        if start != stop:
            if self.requirements[start].nbytes <= self.requirements[stop].nbytes:
                self.insert_copy(slice(start, start))
            else:
                self.insert_copy(slice(stop, stop))
            return
        new_buffer = self.requirements[start].copy()
        new_buffer.aligned = True
        self.requirements.insert(start + 1, new_buffer)
        self.operators.insert(start, core._C)
        self.groups.insert(self.get_operator_group(start), 1)
        # XXX propagate strides

    def get_operator_group(self, ioperator):
        """
        Return operator's group number.

        """
        ilast = 0
        for i, g in enumerate(self.groups):
            last += g
            if ioperator < ilast:
                return i
        return i + 1

    def get_group_slice(self, igroup, reverse=False):
        """
        Return the group slice.

        """
        if igroup < 0:
            igroup = len(self.groups) + igroup
        if igroup >= len(self.groups):
            raise ValueError("Invalid group index '{}'.".format(igroup))
        start = 0
        i = 0
        while True:
            stop = start + self.groups[i]
            if i == igroup:
                break
            i += 1
            start = stop
        if not reverse:
            return slice(start, stop)
        else:
            return slice(stop - 1, start - 1, -1)

    def iter_groups(self, reverse=False):
        if not reverse:
            for igroup in range(self.groups):
                s = self.get_group_slice(igroup)
                yield self.operators[s], self.requirements[1:][s]
            return
        for igroup in range(self.groups, -1, -1):
            s = self.get_group_slice(igroup, reverse=True)
            yield self.operators[s], self.requirements[:-1][s]

    def get_output(self):
        """
        Return the output array, as specified by the last buffer.

        """
        buf = self.requirements[-1]
        return empty(buf.shape, buf.dtype, buf.strides)


class CallingSequenceManager(object):
    """
    This class handles the caching of the different calling sequences
    of an operator.

    """

    def __init__(self, operators):
        self.operators = operators
        self.signatures = {}

    def __call__(self, input, output, operation, preserve_input):
        input = np.asanyarray(input)
        infoin = self.get_info(input)
        infoout = self.get_info(output)
        alias = output is not None and isalias(input, output)
        preserve_input = preserve_input and not alias
        preserve_output = operation is not operation_assignment
        signature = (infoin, infoout, preserve_input, preserve_output, alias)

        try:
            call = self.signatures[signature]
            if output is None:
                output = call.get_output()
        except KeyError:
            call = CallingSequence(self.operators, *signature)
            self.signatures[signature] = call
            if output is None:
                output = call.get_output()
                infoout = self.get_info(output)
                signature = (infoin, infoout, preserve_input, preserve_output, False)
                self.signatures[signature] = call

        return call(input, output, operation)

    @staticmethod
    def get_info(x):
        if x is None:
            return (None, None, None, True)
        return (
            x.dtype,
            x.shape,
            x.strides,
            x.__array_interface__['data'][0] % MEMORY_ALIGNMENT == 0,
        )
