from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import slangpy as spy

from splatgs import gauss, image


@dataclass
class RenderOptions:
    max_sh_degree: int = 4
    background_color: Sequence[float] = field(default_factory=lambda: [.0, .0, .0])
    alpha_thres: float = 1./255.
    max_alpha: float = 0.99
    min_transmittance : float = 0.0001
    nearFar: Sequence[float] = field(default_factory=lambda: [0.2, 100])
    save_data_for_backprop: bool=False
    collect_data_for_densification: bool=False


@dataclass
class RenderContext:
    projections: gauss.ProjectedGaussians
    instances: gauss.GaussiansInstances
    tiles: image.ScreenTiles
    exponential_resizing: bool

    shader_module: spy.Module
    project: spy.Function
    increment_view_counters: spy.Function
    create_instances_and_keys: spy.Function
    find_tile_ranges: spy.Function
    render: spy.Function

    tile_size: int
    block_size: int
    device: torch.device

    @classmethod
    def from_settings(cls, gaussian_num: int, tile_size: int, block_size: int, slang_module: spy.Module, screensize: Sequence[int] | None=None, exponential_resizing: bool=False, device="cuda"):
        projections = gauss.ProjectedGaussians.from_size(gaussian_num, device)
        tiles = image.ScreenTiles.from_screensize(screensize, tile_size, device) if screensize is not None else None
        instances = gauss.GaussiansInstances.from_size(gaussian_num, device=device)
        
        ctx = cls(projections, instances, tiles, exponential_resizing, slang_module, None, None, None, None, None, tile_size, block_size, device)
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
            self.tiles = image.ScreenTiles.from_screensize(screensize, self.tile_size, self.device)
    
    def allocate_instances(self, gaussian_num: int):
        self.instances.allocate_instances(self.instances.cumulative_counts[gaussian_num-1].item(), self.exponential_resizing)

    def reset_tile_ranges(self):
        self.tiles.ranges.zero_()

    def __create_functions(self):
        self.project = self.shader_module.projectGaussians.constants({"TILE_SIZE":self.tile_size}).call_group_shape(spy.slangpy.Shape(self.block_size))
        self.increment_view_counters = self.shader_module.incrementViewCounters.constants({"TILE_SIZE":self.tile_size}).call_group_shape(spy.slangpy.Shape(self.block_size))
        self.create_instances_and_keys = self.shader_module.createGaussianInstances.constants({"TILE_SIZE":self.tile_size}).call_group_shape(spy.slangpy.Shape(self.block_size))
        self.find_tile_ranges = self.shader_module.findTileRanges.constants({"TILE_SIZE":self.tile_size}).call_group_shape(spy.slangpy.Shape(self.block_size))
        self.render = self.shader_module.renderGaussians.constants({"TILE_SIZE":self.tile_size}).call_group_shape(spy.slangpy.Shape(self.tile_size, self.tile_size))


@dataclass
class BackpropagationData:
    accumulated_transmittances: torch.Tensor
    processed_gaussian_counts: torch.Tensor
    are_colors_clamped: torch.Tensor

    @classmethod
    def dummy(cls, device="cuda"):
        dummy_float = torch.zeros(1, dtype=torch.float32, device=device)
        dummy_int = torch.zeros(1, dtype=torch.int32, device=device)
        dummy_bool = torch.zeros(1, dtype=torch.bool, device=device)

        return cls(dummy_float, dummy_int, dummy_bool)

    @classmethod
    def from_settings(cls, gaussian_count: int, screen_size: Sequence[int] | None=None, device="cuda"):
        backprop_data = cls.dummy(device=device)
        if screen_size is None:
            backprop_data.are_colors_clamped = torch.tensor((gaussian_count, 3), dtype=torch.bool, device=device)
        else:
            backprop_data.ensure_capacity(gaussian_count, screen_size)

        return backprop_data

    def ensure_capacity(self, gaussian_count: int, screen_size: Sequence[int]):
        if (self.accumulated_transmittances.shape[0] < screen_size[0]) or (self.accumulated_transmittances.shape[1] < screen_size[1]):
            self.accumulated_transmittances = torch.zeros((screen_size[1], screen_size[0]), dtype=torch.float32, device=self.accumulated_transmittances.device)
        if (self.processed_gaussian_counts.shape[0] < screen_size[0]) or (self.processed_gaussian_counts.shape[1] < screen_size[1]):
            self.processed_gaussian_counts = torch.zeros((screen_size[1], screen_size[0]), dtype=torch.int32, device=self.processed_gaussian_counts.device)

        if self.are_colors_clamped.shape[0] < gaussian_count:
            self.are_colors_clamped = torch.zeros((gaussian_count, 3), dtype=torch.bool, device=self.are_colors_clamped.device)


@dataclass
class DensificationData:
    gaussian_image_radii: torch.Tensor
    view_counters: torch.Tensor

    @classmethod
    def dummy(cls, device="cuda"):
        dummy_float = torch.zeros(1, dtype=torch.float32, device=device)
        dummy_int = torch.zeros(1, dtype=torch.int32, device=device)

        return cls(dummy_float, dummy_int)

    @classmethod
    def from_settings(cls, gaussian_count: int, device="cuda"):
        densif_data = cls.dummy(device=device)
        densif_data.ensure_capacity(gaussian_count=gaussian_count)

        return densif_data

    def ensure_capacity(self, gaussian_count: int):
        if self.gaussian_image_radii.shape[0] < gaussian_count:
            self.gaussian_image_radii = torch.zeros(gaussian_count, dtype=torch.float32, device=self.gaussian_image_radii.device)
        if self.view_counters.shape[0] < gaussian_count:
            self.view_counters = torch.zeros(gaussian_count, dtype=torch.int32, device=self.view_counters.device)

    def zero(self):
        self.gaussian_image_radii.zero_()
        self.view_counters.zero_()


def nearest_multiple(values: np.ndarray, multiple: int):
    return (values + multiple - 1) // multiple * multiple


def render(gs: gauss.Gaussians3D, cam: image.Camera, ctx: RenderContext, opts: RenderOptions=None, image: torch.Tensor | None=None, backprop_data: BackpropagationData | None=None, densif_data: DensificationData | None=None):
    opts = opts if isinstance(opts, RenderOptions) else RenderOptions()
    image = image if isinstance(image, torch.Tensor) else torch.empty((cam.screensize[1], cam.screensize[0], 3), dtype=torch.float32, device=ctx.device)
    backprop_data = backprop_data if isinstance(backprop_data, BackpropagationData) else BackpropagationData.dummy()
    densif_data = densif_data if isinstance(densif_data, DensificationData) else DensificationData.dummy()
    if image.shape[0] < cam.screensize[1] or image.shape[1] < cam.screensize[0] or image.shape[2] < 3:
        raise ValueError(f"Image shape should be at least {(cam.screensize[1], cam.screensize[0], 3)}, was {image.shape}")
    if opts.save_data_for_backprop:
        backprop_data.ensure_capacity(gs.num, cam.screensize)
    if opts.collect_data_for_densification:
        densif_data.ensure_capacity(gs.num)
    ctx.ensure_capacity(gs.num, cam.screensize)

    if opts.collect_data_for_densification:
        ctx.projections.means = ctx.projections.means.detach()
    ctx.project(spy.grid((gs.num,)),
                gs.means, gs.rotations, gs.scales, gs.sh_coefficients,
                ctx.tiles.tiles_xy, cam.intrinsics, cam.half_fov_sin_cos, cam.extrinsics, opts.nearFar,
                gs.use_scale_exponential, gs.color_bias, opts.max_sh_degree, opts.save_data_for_backprop, opts.collect_data_for_densification,
                ctx.projections.means, ctx.projections.depths, ctx.projections.covariances, ctx.projections.colors, ctx.instances.counts,
                backprop_data.are_colors_clamped, densif_data.gaussian_image_radii)
    if opts.collect_data_for_densification:
        ctx.increment_view_counters(spy.grid((gs.num,)), gs.num, densif_data.view_counters, ctx.instances.counts)
        ctx.projections.means.retain_grad()

    torch.cumsum(ctx.instances.counts[:gs.num], dim=0, out=ctx.instances.cumulative_counts[:gs.num])
    ctx.allocate_instances(gs.num)
    if ctx.instances.num > 0:
        ctx.reset_tile_ranges()
        ctx.create_instances_and_keys(spy.grid((gs.num,)), gs.num,
                                    ctx.instances.counts, ctx.instances.cumulative_counts,
                                    ctx.projections.means, ctx.projections.depths, ctx.projections.covariances,
                                    ctx.tiles.tiles_xy,
                                    ctx.instances.instances, ctx.instances.keys)

        torch.sort(ctx.instances.keys[:ctx.instances.num], stable=True, dim=0, out=(ctx.instances.sorted_keys[:ctx.instances.num], ctx.instances.sorted_keys_indices[:ctx.instances.num]))
        torch.index_select(ctx.instances.instances[:ctx.instances.num], dim=0, index=ctx.instances.sorted_keys_indices[:ctx.instances.num], out=ctx.instances.sorted_instances[:ctx.instances.num])
        ctx.find_tile_ranges(spy.grid((ctx.instances.num,)), ctx.instances.num, ctx.instances.sorted_keys, ctx.tiles.ranges)

    render_grid = nearest_multiple(np.array(cam.screensize), ctx.tile_size).tolist()
    ctx.render(spy.grid(render_grid), spy.thread_id(),
               [*cam.screensize], ctx.tiles.tiles_xy,
               ctx.instances.sorted_instances, ctx.tiles.ranges,
               ctx.projections.means, ctx.projections.covariances, ctx.projections.colors, gs.opacities,
               gs.use_opacity_sigmoid, opts.alpha_thres, opts.max_alpha, opts.min_transmittance, opts.save_data_for_backprop, opts.background_color,
               image, backprop_data.accumulated_transmittances, backprop_data.processed_gaussian_counts)

    return image, backprop_data, densif_data
