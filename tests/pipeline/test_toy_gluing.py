import numpy as np
import pytest

from kubo.config import GridConfig, PhysicsConfig
from kubo.grids import build_delta_z_kz_grids_fft
from kubo.greens import realspace_greens_retarded, _fourier_kz_to_z, _build_fft_Gkz_input_for_fixed_omega_kpar

#def _realspace_greens_callable()