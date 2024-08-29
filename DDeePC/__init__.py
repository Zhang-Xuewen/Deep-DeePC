"""
Name: __init__.py
Author: Xuewen Zhang
Date:at 19/04/2024
version: 1.0.0
Description: Import the required modules for easy use.
"""

# Version for DDeePC.
__version__ = "1.0.0"

# Add modules and some specific functions.
# from . import DeePCtools
from .train import train
from .deepc import deepc
from .model import network
from .control import control
from .con_opt import con_opt
from .MyTool import MyTool, NNtool, timer
# from .Plants.waste_water_system import waste_water_system_tensor, waste_water_system
from .Plants.three_tanks import three_tank_system
# from .Plants.siso import siso
# from .Plants.grn import grn
