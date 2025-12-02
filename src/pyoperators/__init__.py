"""
The PyOperators package contains the following modules or packages:

- core : defines the Operator class
- linear : defines standard linear operators
- nonlinear : defines non-linear operators (such as thresholding or rounding)
- iterative : defines iterative algorithms working with operators
- utils : miscellaneous routines
- operators_mpi : MPI operators (even if mpi4py is not present)
- operators_pywt : (optional) loaded if PyWavelets is present.

"""

from importlib.metadata import version as _version

from .core import (
    AdditionOperator,
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    BlockSliceOperator,
    CompositionOperator,
    ConstantOperator,
    DiagonalOperator,
    GroupOperator,
    HomothetyOperator,
    IdentityOperator,
    MultiplicationOperator,
    Operator,
    ReductionOperator,
    ReshapeOperator,
    Variable,
    ZeroOperator,
    asoperator,
    timer_operator,
)
from .fft import ConvolutionOperator, FFTOperator
from .iterative import pcg
from .linear import (
    BandOperator,
    DegreesOperator,
    DenseBlockDiagonalOperator,
    DenseOperator,
    DiagonalNumexprOperator,
    DifferenceOperator,
    EigendecompositionOperator,
    IntegrationTrapezeOperator,
    MaskOperator,
    PackOperator,
    RadiansOperator,
    Rotation2dOperator,
    Rotation3dOperator,
    SparseOperator,
    SumOperator,
    SymmetricBandOperator,
    SymmetricBandToeplitzOperator,
    TridiagonalOperator,
    UnpackOperator,
)
from .nonlinear import (
    Cartesian2SphericalOperator,
    ClipOperator,
    HardThresholdingOperator,
    MaximumOperator,
    MaxOperator,
    MinimumOperator,
    MinMaxOperator,
    MinOperator,
    NormalizeOperator,
    NumexprOperator,
    PowerOperator,
    ProductOperator,
    ReciprocalOperator,
    RoundOperator,
    SoftThresholdingOperator,
    Spherical2CartesianOperator,
    SqrtOperator,
    SquareOperator,
    To1dOperator,
    ToNdOperator,
)
from .operators_mpi import (
    MPIDistributionGlobalOperator,
    MPIDistributionIdentityOperator,
)
from .operators_pywt import Wavelet2dOperator, WaveletOperator
from .proxy import proxy_group
from .rules import rule_manager
from .utils import operation_assignment
from .utils.mpi import MPI
from .warnings import PyOperatorsWarning

__all__ = [
    'AdditionOperator',
    'BandOperator',
    'BlockColumnOperator',
    'BlockDiagonalOperator',
    'BlockRowOperator',
    'BlockSliceOperator',
    'Cartesian2SphericalOperator',
    'ClipOperator',
    'CompositionOperator',
    'ConstantOperator',
    'ConvolutionOperator',
    'DegreesOperator',
    'DenseBlockDiagonalOperator',
    'DenseOperator',
    'DiagonalNumexprOperator',
    'DiagonalOperator',
    'DifferenceOperator',
    'EigendecompositionOperator',
    'FFTOperator',
    'GroupOperator',
    'HardThresholdingOperator',
    'HomothetyOperator',
    'IdentityOperator',
    'IntegrationTrapezeOperator',
    'MPIDistributionGlobalOperator',
    'MPIDistributionIdentityOperator',
    'MaskOperator',
    'MaxOperator',
    'MaximumOperator',
    'MinMaxOperator',
    'MinOperator',
    'MinimumOperator',
    'MultiplicationOperator',
    'NormalizeOperator',
    'NumexprOperator',
    'Operator',
    'PackOperator',
    'PowerOperator',
    'ProductOperator',
    'PyOperatorsWarning',
    'RadiansOperator',
    'ReciprocalOperator',
    'ReductionOperator',
    'ReshapeOperator',
    'Rotation2dOperator',
    'Rotation3dOperator',
    'RoundOperator',
    'SoftThresholdingOperator',
    'SparseOperator',
    'Spherical2CartesianOperator',
    'SqrtOperator',
    'SquareOperator',
    'SumOperator',
    'SymmetricBandOperator',
    'SymmetricBandToeplitzOperator',
    'To1dOperator',
    'ToNdOperator',
    'TridiagonalOperator',
    'UnpackOperator',
    'Variable',
    'Wavelet2dOperator',
    'WaveletOperator',
    'ZeroOperator',
    'asoperator',
    'I',
    'MPI',
    'O',
    'operation_assignment',
    'pcg',
    'proxy_group',
    'rule_manager',
    'timer_operator',
    'X',
]

I = IdentityOperator()  # noqa: E741
O = ZeroOperator()  # noqa: E741
X = Variable('X')

__version__ = _version('pyoperators')
