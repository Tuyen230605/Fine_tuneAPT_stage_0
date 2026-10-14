import os
from omni.isaac.core import World
import omni.isaac.core.utils.prims as prims_utils
import omni.isaac.core.utils.string as strings_utils
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.objects import GroundPlane
from typing import Union, List, Optional, Sequence
from pxr import UsdGeom, Gf, Sdf, UsdShade, Usd

def add_ground_plane(
    my_world: World, 
    usd_path="assets/Grid4.1/default_environment.usd", 
    z_position: float = 0,
    name="default_ground_plane",
    prim_path: str = "/World/defaultGroundPlane",
    static_friction: float = 0.5,
    dynamic_friction: float = 0.5,
    restitution: float = 0.8,
):
    prims_utils.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    if "default_environment" in usd_path:
        usd_path_rel2abs(usd_path, prim_path)
    else:
        loaded_usd_texture_path_rel2abs(prim_path)

    physics_material_path = strings_utils.find_unique_string_name(
        initial_name="/World/Physics_Materials/physics_material", 
        is_unique_fn=lambda x: not prims_utils.is_prim_path_valid(x)
    )
    physics_material = PhysicsMaterial(
        prim_path=physics_material_path,
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )
    plane = GroundPlane(prim_path=prim_path, name=name, z_position=z_position, 
                        physics_material=physics_material)
    my_world.scene.add(plane)


def usd_path_rel2abs(usd_path: str, ground_prim_path: str):
    usd_dir = os.path.dirname(usd_path)
    attrs = ["diffuse_texture", "emissive_color_texture", "emissive_mask_texture"]
    # attrs = ["diffuse_texture"]
    shader_prim = prims_utils.get_prim_at_path(ground_prim_path + "/Looks/theGrid/Shader")
    for attr_name in attrs:
        attr = shader_prim.GetAttribute(f"inputs:{attr_name}")
        usd_path = str(attr.Get()).replace("@", "")
        if usd_path.startswith("Materials"):
            attr.Set(usd_dir + "/" + usd_path)
        if usd_path.startswith("./"):
            attr.Set(usd_dir + "/" + usd_path[2:])


def traverse_shaders(root_prim: Union[str, Usd.Prim]):
    if isinstance(root_prim, str):
        root_prim = prims_utils.get_prim_at_path(root_prim)
    
    results: List[Usd.Prim] = []
    iterator = iter(Usd.PrimRange(root_prim))
    for prim in iterator:
        if prim.IsA(UsdShade.Shader):
            results.append(prim)
    return results


def get_reference_usd(prim: Union[str, Usd.Prim]):
    if isinstance(prim, str):
        prim = prims_utils.get_prim_at_path(prim)
    
    references: List[Sdf.Reference] = []
    for prim_spec in prim.GetPrimStack():
        references.extend(prim_spec.referenceList.prependedItems)
    return references

# for objects in Axis_Aligned folder
def texture_path_rel2abs(usd_dir: str, shader: Usd.Prim, asset_path: str=None):
    attrs = ["diffuse_texture", "emissive_mask_texture"]

    for attr_name in attrs:
        attr = shader.GetAttribute(f"inputs:{attr_name}")
        if asset_path is None:
            attr_val = attr.Get()
            if isinstance(attr_val, Sdf.AssetPath):
                # print(f"name: {attr_name}, path: {attr_val.path}, rpath: {attr_val.resolvedPath}")
                asset_path: str = attr_val.path
                if asset_path.startswith("Materials"):
                    attr.Set(usd_dir + "/" + asset_path)
                if asset_path.startswith("./"):
                    attr.Set(usd_dir + "/" + asset_path[2:])
        else:
            attr.Set(asset_path)


def loaded_usd_texture_path_rel2abs(prim: Union[str, Usd.Prim]):
    """IsaacSim 4.1 behaves different with 2023.1.1 when loading .usd file 
    converted from .obj file, this is strange. This is a temporal fix.
    """
    if isinstance(prim, str):
        prim = prims_utils.get_prim_at_path(prim)
    
    ref_usd_path = get_reference_usd(prim)[0].assetPath
    ref_usd_dir = os.path.dirname(ref_usd_path)

    asset_path = None
    if "gridroom_black" in ref_usd_path:
        asset_path = "assets/mesh/grid/Materials/Textures/WireframeBlur_basecolor.png"

    shader_prims = traverse_shaders(prim)
    for shader in shader_prims:
        texture_path_rel2abs(ref_usd_dir, shader, asset_path)