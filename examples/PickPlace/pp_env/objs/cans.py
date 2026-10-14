"""Soda-can object classes (USDs from assets/mesh/cans_usd/).

All cans share the same physical scale; we factor it out as `CAN_SCALE` and
use the reference `ObjBase` (cylindrical grasp) implementation unchanged.
"""

import numpy as np

from .base import ObjBase


# Uniform scale applied to every soda-can USD/OBJ.
CAN_SCALE = np.ones(3) * 0.15


class Can7up01(ObjBase):
    USD_PATH = "assets/mesh/cans_usd/7up01/single.usd"
    OBJ_PATH = "assets/mesh/cans_std/7up01/single.obj"
    SCALE = CAN_SCALE


class Can7up04(ObjBase):
    USD_PATH = "assets/mesh/cans_usd/7up04/single.usd"
    OBJ_PATH = "assets/mesh/cans_std/7up04/single.obj"
    SCALE = CAN_SCALE


class CanCoke01(ObjBase):
    USD_PATH = "assets/mesh/cans_usd/coke01/single.usd"
    OBJ_PATH = "assets/mesh/cans_std/coke01/single.obj"
    SCALE = CAN_SCALE


class CanFanta01(ObjBase):
    USD_PATH = "assets/mesh/cans_usd/fanta01/single.usd"
    OBJ_PATH = "assets/mesh/cans_std/fanta01/single.obj"
    SCALE = CAN_SCALE


class CanPepsi01(ObjBase):
    USD_PATH = "assets/mesh/cans_usd/pepsi01/single.usd"
    OBJ_PATH = "assets/mesh/cans_std/pepsi01/single.obj"
    SCALE = CAN_SCALE


class CanPepsi02(ObjBase):
    USD_PATH = "assets/mesh/cans_usd/pepsi02/single.usd"
    OBJ_PATH = "assets/mesh/cans_std/pepsi02/single.obj"
    SCALE = CAN_SCALE


class CanPepsi02a(ObjBase):
    USD_PATH = "assets/mesh/cans_usd/pepsi02a/single.usd"
    OBJ_PATH = "assets/mesh/cans_std/pepsi02a/single.obj"
    SCALE = CAN_SCALE


class CanSprite01(ObjBase):
    USD_PATH = "assets/mesh/cans_usd/sprite01/single.usd"
    OBJ_PATH = "assets/mesh/cans_std/sprite01/single.obj"
    SCALE = CAN_SCALE
