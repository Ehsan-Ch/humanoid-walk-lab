"""A planar humanoid gait-learning experiment in MuJoCo."""
import os

# Optional process-local path to a trusted Microsoft C++ runtime on Windows.
# Most standard Python installations already have the redistributable installed.
_dll_handle = None
if os.name == 'nt' and os.environ.get('WALK_LAB_DLL_DIR'):
    _dll_handle = os.add_dll_directory(os.environ['WALK_LAB_DLL_DIR'])
