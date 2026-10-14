import omni
import inspect
# core
import omni.kit.app
from omni.isaac.core.objects import cuboid
from omni.isaac.core.materials.omni_glass import OmniGlass
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims import RigidPrim, GeometryPrim
from omni.isaac.core.utils.semantics import add_update_semantics
import omni.isaac.core.utils.prims as prims_utils
import omni.isaac.core.utils.stage as stage_utils
# usd core
import omni.usd
from pxr import UsdGeom, Gf, Sdf, UsdShade, Usd

import os
import math
import PIL.Image
import numpy as np
from typing import Union, List, Optional, Sequence


def add_object(
    usd_path: str, 
    prim_path: str, 
    name: str,
    position: Optional[np.ndarray] = None,
    translation: Optional[np.ndarray] = None,
    orientation: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
    visible: Optional[bool] = None,
    collision: Optional[bool] = None, 
    approx: str = "none",
    fixed: bool = False,
    mass: Optional[float] = None,
    density: Optional[float] = None,
    linear_velocity: Optional[Sequence[float]] = None,
    angular_velocity: Optional[Sequence[float]] = None,
    **kwargs
):
    stage_utils.add_reference_to_stage(usd_path, prim_path)
    loaded_usd_texture_path_rel2abs(prim_path)

    return UsdObject(
        prim_path=prim_path,
        name=name,
        position=position,
        translation=translation,
        orientation=orientation,
        scale=scale,
        visible=visible,
        collision=collision,
        approx=approx,
        fixed=fixed,
        mass=mass,
        density=density,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        **kwargs
    )


class UsdObject(RigidPrim, GeometryPrim):
    def __init__(
        self,
        prim_path: str,
        name: str = "obj",
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        collision: Optional[bool] = None, 
        approx: str = "none",
        fixed: bool = False,
        mass: Optional[float] = None,
        density: Optional[float] = None,
        linear_velocity: Optional[Sequence[float]] = None,
        angular_velocity: Optional[Sequence[float]] = None,
        **kwargs
    ) -> None:
        geo_req_kwargs = inspect.signature(GeometryPrim).parameters
        geo_get_kwargs = {k:v for k, v in kwargs.items() if k in geo_req_kwargs}
        print("GeometryPrim kwargs:"); print(geo_get_kwargs)
        GeometryPrim.__init__(
            self,
            prim_path=prim_path, 
            name=name, 
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            collision=collision, 
            **geo_get_kwargs)
        if collision:
            self.set_collision_approximation(approx)
        
        self.is_rigid = not fixed
        if not fixed:
            RigidPrim.__init__(
                self,
                prim_path=prim_path,
                name=name,
                position=position,
                translation=translation,
                orientation=orientation,
                scale=scale,
                visible=visible,
                mass=mass,
                density=density,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity
            )
            # self.set_sleep_threshold(0)


def add_hdr_light(
    prim_path: str,
    hdr_file: str, 
    intensity=1000,
):
    # lightining
    light = prims_utils.create_prim(
        prim_path= prim_path,
        prim_type= 'DomeLight', 
        attributes={
            'inputs:intensity': intensity, 
            'inputs:texture:format': 'latlong',
            'inputs:texture:file': hdr_file
        }
    )
    return light


def add_plane_mesh(
    H: int,
    W: int, 
    prim_path: str, 
    scale = 1.0, 
):
    # https://openusd.org/release/tut_simple_shading.html#simple-shading-in-usd
    stage = stage_utils.get_current_stage()
    mesh: UsdGeom.Mesh = UsdGeom.Mesh.Define(stage, prim_path)

    vertices = np.array([
        [-1, -1, 0], [1, -1, 0],
        [-1, 1, 0], [1, 1, 0]
    ]).astype(np.float64) / 2 * scale

    vertices[:, 0] *= W
    vertices[:, 1] *= H
    vertices = vertices / math.sqrt(H*W)

    faces = np.array([[0, 1, 2], [2, 1, 3]])
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    mesh.CreatePointsAttr(vertices.astype(np.float32).tolist())
    mesh.CreateFaceVertexCountsAttr([3] * faces.shape[0])
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    mesh.CreateNormalsAttr([(0, 0, 1)] * vertices.shape[0])

    texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
    )
    texCoords.Set(uv.tolist())
    return mesh


def add_image_material(
    image_path: str,
    prim_path: str,
    roughness=0.5,
    metallic=0.0
):
    # https://openusd.org/release/tut_simple_shading.html#simple-shading-in-usd
    stage = stage_utils.get_current_stage()
    material: UsdShade.Material = UsdShade.Material.Define(stage, prim_path)

    pbrShader: UsdShade.Shader = UsdShade.Shader.Define(stage, prim_path + "/PBRShader")
    pbrShader.CreateIdAttr("UsdPreviewSurface")
    pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

    material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

    stReader: UsdShade.Shader = UsdShade.Shader.Define(stage, prim_path + "/stReader")
    stReader.CreateIdAttr("UsdPrimvarReader_float2")

    diffuseTextureSampler: UsdShade.Shader = UsdShade.Shader.Define(
        stage, prim_path + "/diffuseTexture")
    diffuseTextureSampler.CreateIdAttr("UsdUVTexture")
    diffuseTextureSampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(image_path)
    diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        stReader.ConnectableAPI(), "result")
    diffuseTextureSampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        diffuseTextureSampler.ConnectableAPI(), "rgb")

    stInput = material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
    stInput.Set("st")

    stReader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(stInput)
    return material


def add_image_plane(
    image_path: str, 
    prim_path: str, 
    scale = 1.0, 
    roughness = 0.5,
    metallic = 0.0,
    force_square = False
):
    image = PIL.Image.open(image_path)
    H, W = image.height, image.width
    if force_square:
        H = W = max(H, W)

    root_path = prim_path
    mesh_path = prim_path + "/mesh"
    material_path = prim_path + "/material"

    mesh = add_plane_mesh(H, W, mesh_path, scale)
    material = add_image_material(image_path, material_path,
                                  roughness, metallic)

    mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(mesh).Bind(material)
    return root_path, mesh_path, material_path


def change_shader_texture(
    look_prefix: str,
    texture_path: str
):
    grid_look = prims_utils.get_prim_at_path(
        prim_path="/World/defaultGroundPlane/Looks/theGrid/Shader"
    )
    grid_look.GetAttribute("inputs:diffuse_texture").Set(texture_path)
    grid_look.GetAttribute("inputs:emissive_color_texture").Set(texture_path)
    grid_look.GetAttribute("inputs:emissive_mask_texture").Set(texture_path)

    stage = stage_utils.get_current_stage()
    lp = stage.GetPrimAtPath(look_prefix)
    mtl_prims = lp.GetAllChildren()
    shd_prims = [p.GetPrimPath().pathString + '/Shader' for p in mtl_prims]
    selection = omni.usd.get_context().get_selection()
    selection.set_selected_prim_paths(shd_prims, False)
    app = omni.kit.app.get_app()
    for _ in range(5):
        app.update()

    grid_look.GetAttribute("inputs:texture_translate").Set(Gf.Vec2f(0.0, 4.0))
    grid_look.GetAttribute("inputs:texture_scale").Set(Gf.Vec2f(0.0, 0.0))


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
def texture_path_rel2abs(usd_dir: str, shader: Usd.Prim):
    attrs = [
        "diffuse_texture", 
        "reflectionroughness_texture",
        "ao_texture",
        "emissive_color_texture", 
        "emissive_mask_texture",
        "opacity_texture",
        "normalmap_texture",
        "detail_normalmap_texture",
        "ORM_texture"
    ]
    # attrs = ["diffuse_texture"]
    for attr_name in attrs:
        attr = shader.GetAttribute(f"inputs:{attr_name}")
        attr_val = attr.Get()
        if isinstance(attr_val, Sdf.AssetPath):
            # print(f"name: {attr_name}, path: {attr_val.path}, rpath: {attr_val.resolvedPath}")
            asset_path: str = attr_val.path
            if asset_path.startswith("Materials"):
                attr.Set(usd_dir + "/" + asset_path)
            elif asset_path.startswith("./"):
                attr.Set(usd_dir + "/" + asset_path[2:])


# for objects in Axis_Aligned folder
def modify_texture_path_rel2abs(usd_dir: str, shader: Usd.Prim):
    attrs = [
        "diffuse_texture", 
        "reflectionroughness_texture",
        "ao_texture",
        "emissive_color_texture", 
        "emissive_mask_texture",
        "opacity_texture",
        "normalmap_texture",
        "detail_normalmap_texture",
        "ORM_texture"
    ]
    # attrs = ["diffuse_texture"]

    new_material_paths = [
        # "Materials/Textures/basic_block_blue1_BaseColor.png",
        # "Materials/Textures/basic_block_green1_BaseColor.png",
        # "Materials/Textures/basic_block_red1_BaseColor.png",
        # "Materials/Textures/basic_block_yellow_BaseColor.png",
        "Materials/Textures/material_0.png"
    ]
    new_material_path = np.random.choice(new_material_paths)

    for attr_name in attrs:
        attr = shader.GetAttribute(f"inputs:{attr_name}")
        attr_val = attr.Get()
        if isinstance(attr_val, Sdf.AssetPath):
            # print(f"name: {attr_name}, path: {attr_val.path}, rpath: {attr_val.resolvedPath}")
            asset_path: str = attr_val.path
            if asset_path.startswith("Materials"):
                attr.Set(usd_dir + "/" + new_material_path)
            elif asset_path.startswith("./"):
                attr.Set(usd_dir + "/" + new_material_path)
            elif "Materials" in asset_path:
                attr.Set(usd_dir + "/" + new_material_path)
    
    # to save as new usd file with new name
    stage = stage_utils.get_current_stage()
    new_usd_path = os.path.join(os.path.dirname(usd_dir), new_material_path.split("/")[-1].split(".")[0] + ".usd")
    stage.Export(new_usd_path)

def loaded_usd_texture_path_rel2abs(prim: Union[str, Usd.Prim]):
    """IsaacSim 4.1 behaves different with 2023.1.1 when loading .usd file 
    converted from .obj file, this is strange. This is a temporal fix.
    """
    if isinstance(prim, str):
        prim = prims_utils.get_prim_at_path(prim)
    
    ref_usd_path = get_reference_usd(prim)[0].assetPath
    ref_usd_dir = os.path.dirname(ref_usd_path)
    
    shader_prims = traverse_shaders(prim)
    for shader in shader_prims:
        texture_path_rel2abs(ref_usd_dir, shader)

def loaded_modify_usd_texture_path_rel2abs(prim: Union[str, Usd.Prim]):
    """IsaacSim 4.1 behaves different with 2023.1.1 when loading .usd file 
    converted from .obj file, this is strange. This is a temporal fix.
    """
    if isinstance(prim, str):
        prim = prims_utils.get_prim_at_path(prim)
    
    ref_usd_path = get_reference_usd(prim)[0].assetPath
    ref_usd_dir = os.path.dirname(ref_usd_path)
    
    shader_prims = traverse_shaders(prim)
    for shader in shader_prims:
        modify_texture_path_rel2abs(ref_usd_dir, shader)

