"""Object registry for pick-place scene generation.

Each class binds a USD/OBJ asset to a `GraspProposal` subclass. The eval scripts
look up classes by name via `getattr(objs, "<ClassName>")` based on the
`pickable_objs` list in `eval/configs/<setting>.yaml`.

Categories (grouped by source mesh directory):
    cans       7 soda cans          (assets/mesh/cans_usd/)
    blocks     4 colored cubes      (assets/mesh/blocks/)
    ycb        3 YCB-style objects  (assets/mesh/Axis_Aligned/)
    graspnet   21 GraspNet objects  (assets/mesh/GraspNet_models/)

To add a new object class:
    1. Drop the USD into the appropriate assets/mesh/<dir>/ directory.
    2. Add the class to objs/<category>.py (subclass `ObjBase` and set
       USD_PATH / OBJ_PATH / SCALE, optionally overriding grasp logic).
    3. Re-export it below.
    4. Reference it from the relevant eval/configs/*.yaml.
"""

from . import base

from .cans import (
    Can7up01, Can7up04, CanCoke01, CanFanta01,
    CanPepsi01, CanPepsi02, CanPepsi02a,
    CanSprite01,
)
from .ycb import (
    CanTomatoSoup, WoodenBlock, Clamp,
)
from .blocks import (
    BlueBlock, GreenBlock, RedBlock, YellowBlock,
)
from .graspnet import (
    Banana, PowerDrill,
    Apple, Pear, Orange, WhiteCup, ToyAirplane, ToothPaste,
    Zebra, Rhinocero, YellowDrink, DarlieBox, Soap,
    Toy01, Toy02, BlueBall, Toy05,
    Camel, Elephant, Toy03, Toy04,
)

__all__ = [
    "base", "Can7up01",
    "Can7up04", "CanCoke01", "CanFanta01",
    "CanPepsi01", "CanPepsi02", "CanPepsi02a", "CanSprite01",
    "CanTomatoSoup", "WoodenBlock", "Clamp",
    "BlueBlock", "GreenBlock", "RedBlock", "YellowBlock",
    "Banana", "PowerDrill",
    "Apple", "Pear", "Orange", "WhiteCup", "ToyAirplane", "ToothPaste",
    "Zebra", "Rhinocero", "YellowDrink", "DarlieBox", "Soap",
    "Toy01", "Toy02", "BlueBall", "Toy05",
    "Camel", "Elephant", "Toy03", "Toy04",
]
