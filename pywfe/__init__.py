import json
import os
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


PACKAGE_DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database")
DEFAULT_USER_DATABASE_PATH = os.path.join(
    os.path.expanduser("~"), ".pywfe", "database")

# Path to a config file to store the user's preferred database location
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".pywfe_config.json")


def get_user_database_path():
    """Retrieve the user database path, falling back to default if not set."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            return config.get("USER_DATABASE_PATH", DEFAULT_USER_DATABASE_PATH)
        except json.JSONDecodeError:
            print("Warning: Config file is corrupted. Using default database path.")
    return DEFAULT_USER_DATABASE_PATH


def set_user_database_path(new_path):
    """Update and persist the user database path."""
    os.makedirs(new_path, exist_ok=True)  # Ensure the new directory exists
    config = {"USER_DATABASE_PATH": new_path}
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)

    global USER_DATABASE_PATH
    USER_DATABASE_PATH = new_path

    print(f"User database path updated to: {new_path}")


# Load the user database path (default or overridden)
USER_DATABASE_PATH = get_user_database_path()
