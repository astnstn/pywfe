"""pywfe. A Python implementation of the WFE method"""

from pywfe.model import Model
from pywfe.core.eigensolvers import solver
from pywfe.utils.comsol_loader import load_comsol
from pywfe.utils.modal_assurance import sort_wavenumbers
from pywfe.utils.io_utils import save, load, database
from pywfe.utils.forcer import Forcer
from pywfe.utils.vtk_tools import sort_to_vtk, save_as_vtk
from pywfe.utils.shaker import Shaker
from pywfe.utils.logging_config import log
from pywfe.utils.trf_interpolator import interpolate
from pywfe.utils.pipe_fluid_structure_energy_distribution import calculate_power_distribution

log()

import os
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'database')