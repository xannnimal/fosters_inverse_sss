"""Foster's inverse and noise-weighted SSS comparison tools."""

from .basis import compute_basis
from .methods import foster_operator, foster_prior, standard_sss
from .noise import compute_noise_cov
from .reference import check_against_reference, reference_inverse

__all__ = [
    "check_against_reference",
    "compute_basis",
    "compute_noise_cov",
    "foster_operator",
    "foster_prior",
    "standard_sss",
    "reference_inverse",
]

__version__ = "0.1.0"
