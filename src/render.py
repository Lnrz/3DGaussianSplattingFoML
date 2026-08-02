import data
import torch
from dataclasses import dataclass, field
from collections.abc import Sequence
import slangpy as spy

@dataclass
class Camera:
    screensize: tuple[int,int] | Sequence[int]
    intrinsics: torch.Tensor
    half_fov_sin_cos: torch.Tensor
    extrinsics: torch.Tensor

@dataclass
class RenderOptions:
    max_sh_degree: int = 4
    covariance_determinant_thres: float = 1e-4
    alpha_thres: float = 1./255.
    max_alpha: float = 0.99
    min_transmittance : float = 0.0001
    nearFar: Sequence[float] = field(default_factory=lambda: [0.01, 100])
    save_data_for_backprop: bool=False

@dataclass
class RenderContext:
    projections: data.ProjectedGaussians
    instances: data.GaussiansInstances
    tiles: data.ScreenTiles
    exponential_resizing: bool

    shader_module: spy.Module
    project: spy.Function
    create_instances_and_keys: spy.Function
    find_tile_ranges: spy.Function
    render: spy.Function

    tile_size: int
    block_size: int
    device: any

    dummy_2d_float: torch.Tensor
    dummy_2d_int: torch.Tensor

    @classmethod
    def from_settings(cls, gaussian_num: int, tile_size: int, block_size: int, slang_module: spy.Module, screensize=None, exponential_resizing: bool=False, device="cuda"):
        projections = data.ProjectedGaussians.from_size(gaussian_num, device)
        tiles = data.ScreenTiles.from_screensize(screensize, tile_size, device) if screensize else None
        instances = data.GaussiansInstances.from_size(gaussian_num, device=device)
        
        dummy_2d_float = torch.empty((1,1), dtype=torch.float32, device=device)
        dummy_2d_int = torch.empty_like(dummy_2d_float, dtype=torch.int32)

        ctx = cls(projections, instances, tiles, exponential_resizing, slang_module, None, None, None, None, tile_size, block_size, device, dummy_2d_float, dummy_2d_int)
        ctx.__create_functions()

        return ctx

    def change_shader_size(self, tile_size: int = 0, block_size: int = 0):
        tile_size = tile_size if tile_size > 0 else self.tile_size
        block_size = block_size if block_size > 0 else self.block_size
        if tile_size == self.tile_size and block_size == self.block_size:
            return

        self.tile_size = tile_size
        self.block_size = block_size
        self.__create_functions()

    def ensure_capacity(self, gaussian_num: int, screensize):
        self.projections.ensure_capacity(gaussian_num)
        self.instances.ensure_capacity(gaussian_num)
        if self.tiles is not None:
            self.tiles.ensure_capacity(screensize, self.tile_size)
        else:
            self.tiles = data.ScreenTiles.from_screensize(screensize, self.tile_size, self.device)
    
    def allocate_instances(self, gaussian_num: int):
        self.instances.allocate_instances(self.instances.cumulative_counts[gaussian_num-1].item(), self.exponential_resizing)

    def reset_tile_ranges(self):
        self.tiles.ranges.zero_()

    def __create_functions(self):
        self.project = self.shader_module.projectGaussians.constants({"TILE_SIZE":self.tile_size}).call_group_shape(spy.slangpy.Shape(self.block_size))
        self.create_instances_and_keys = self.shader_module.createGaussianInstances.constants({"TILE_SIZE":self.tile_size}).call_group_shape(spy.slangpy.Shape(self.block_size))
        self.find_tile_ranges = self.shader_module.findTileRanges.constants({"TILE_SIZE":self.tile_size}).call_group_shape(spy.slangpy.Shape(self.block_size))
        self.render = self.shader_module.renderGaussians.constants({"TILE_SIZE":self.tile_size}).call_group_shape(spy.slangpy.Shape(self.tile_size, self.tile_size))


def create_slangpy_device_for_torch(type: spy.DeviceType=spy.DeviceType.cuda, include_paths=[], torch_device=None,
                          fp_mode: spy.SlangFloatingPointMode=spy.SlangFloatingPointMode.default,
                          optimization_level: spy.SlangOptimizationLevel=spy.SlangOptimizationLevel.default):
    torch.cuda.init()
    torch.cuda.current_device()
    torch.cuda.current_stream()
    if torch_device is None:
        torch_device = torch.cuda.current_device()
    with torch.device(torch_device):
        handles = spy.get_cuda_current_context_native_handles()

    return spy.Device(
        type=type,
        compiler_options= {
            "include_paths": [spy.SHADER_PATH] + include_paths,
            "floating_point_mode" : fp_mode,
            "optimization" : optimization_level,
            "disable_warnings" : ["31000"] # warning about shared memory array of link-time constant length being experimental
        },
        enable_cuda_interop=(type != spy.DeviceType.cuda),
        existing_device_handles=handles,
    )

def render(gs: data.Gaussians3D, cam: Camera, ctx: RenderContext, opts: RenderOptions=None, image: torch.Tensor | None=None, backprop_data: tuple[torch.Tensor, torch.Tensor] | None=None):
    if not isinstance(opts, RenderOptions):
        opts = RenderOptions()
    if not isinstance(image, torch.Tensor):
        image = torch.empty((cam.screensize[1], cam.screensize[0], 3), dtype=torch.float32, device=ctx.device)
    if not opts.save_data_for_backprop:
        backprop_data = (ctx.dummy_2d_float, ctx.dummy_2d_int)
    elif not isinstance(backprop_data, tuple[torch.Tensor, torch.Tensor]):
        backprop_data = (
            torch.empty((cam.screensize[1], cam.screensize[0]), dtype=torch.float32, device=ctx.device),
            torch.empty((cam.screensize[1], cam.screensize[0]), dtype=torch.int32, device=ctx.device)
        )
    ctx.ensure_capacity(gs.num, cam.screensize)

    ctx.project(spy.grid((gs.num,)),
                gs.means, gs.rotations, gs.scales, gs.sh_coefficients,
                ctx.tiles.tiles_xy, cam.intrinsics, cam.half_fov_sin_cos, cam.extrinsics, opts.nearFar,
                gs.use_scale_exponential, opts.covariance_determinant_thres, gs.color_bias, opts.max_sh_degree,
                ctx.projections.means, ctx.projections.depths, ctx.projections.covariances, ctx.projections.colors, ctx.instances.counts)
    
    torch.cumsum(ctx.instances.counts[:gs.num], dim=0, out=ctx.instances.cumulative_counts[:gs.num])
    ctx.allocate_instances(gs.num)
    if ctx.instances.num > 0:
        ctx.create_instances_and_keys(spy.grid((gs.num,)), gs.num,
                                    ctx.instances.counts, ctx.instances.cumulative_counts,
                                    ctx.projections.means, ctx.projections.depths, ctx.projections.covariances,
                                    ctx.tiles.tiles_xy,
                                    ctx.instances.instances, ctx.instances.keys)
        
        torch.sort(ctx.instances.keys[:ctx.instances.num], stable=True, dim=0, out=(ctx.instances.sorted_keys[:ctx.instances.num], ctx.instances.sorted_keys_indices[:ctx.instances.num]))
        torch.index_select(ctx.instances.instances[:ctx.instances.num], dim=0, index=ctx.instances.sorted_keys_indices[:ctx.instances.num], out=ctx.instances.sorted_instances[:ctx.instances.num])
        ctx.find_tile_ranges(spy.grid((ctx.instances.num,)), ctx.instances.num, ctx.instances.sorted_keys, ctx.tiles.ranges)
    
    ctx.render(spy.grid(cam.screensize), spy.thread_id(),
               ctx.tiles.tiles_xy,
               ctx.instances.sorted_instances, ctx.tiles.ranges,
               ctx.projections.means, ctx.projections.covariances, ctx.projections.colors, gs.opacities,
               gs.use_opacity_sigmoid, opts.alpha_thres, opts.max_alpha, opts.min_transmittance, opts.save_data_for_backprop,
               image, backprop_data[0], backprop_data[1])

    if ctx.instances.num > 0:
        ctx.reset_tile_ranges()

    return image if not opts.save_data_for_backprop else image, backprop_data