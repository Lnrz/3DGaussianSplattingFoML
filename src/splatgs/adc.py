from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import slangpy as spy

from splatgs import gauss


@dataclass
class AdaptiveDensityControl:
    """Class holding adaptive density control data

    - `accumulated_norms`: Gaussian accumulated gradient norms
    - `prng`: Torch Generator for spawning Gaussians when splitting
    - `block_size`: block size of Slang kernels
    - `shader_module`: adaptive density control Slang module
    - `scale_threshold`: scale threshold above which Gaussians are split and below which Gaussians are cloned
    - `world_radius_threshold`: scale threshold above which Gaussians are pruned
    - `image_radius_threshold`: image radius threshold above which Gaussians are pruned
    - `opacity_threshold`: opacity threshold below which Gaussians are pruned
    - `gradient_threshold`: gradient norm threshold above which Gaussians are densified
    - `split_factor`: factor by which dividing Gaussian scales when splitting
    - `opacity_reset_value`: opacity reset value
    """

    accumulated_norms: torch.Tensor

    prng: torch.Generator

    block_size: int
    shader_module: spy.Module

    accumulate_norms: spy.Function
    count_copies: spy.Function
    create_copies: spy.Function

    scale_threshold: float
    world_radius_threshold: float

    image_radius_threshold: float = 20.
    opacity_threshold: float = 0.005
    gradient_threshold: float = 0.0002
    split_factor: float = 1.6
    opacity_reset_value: float = 0.01

    @classmethod
    def from_settings(cls, gaussian_num: int, shader_module: spy.Module, block_size: int, scale_threshold: float, world_radius_threshold: float,
                      image_radius_threshold: float=20., opacity_threshold: float=0.005, opacity_reset_value: float=0.01, split_factor: float=1.6, gradient_threshold: float=0.0002, seed=42, device="cuda"):
        return cls(torch.zeros(gaussian_num, dtype=torch.float32, device=device),
                   torch.Generator().manual_seed(seed),
                   block_size, shader_module,
                   shader_module.accumulateNorms.call_group_shape(spy.slangpy.Shape(block_size)),
                   shader_module.countCopies.call_group_shape(spy.slangpy.Shape(block_size)),
                   shader_module.createCopies.call_group_shape(spy.slangpy.Shape(block_size)),
                   scale_threshold,
                   world_radius_threshold, image_radius_threshold, opacity_threshold,
                   gradient_threshold, split_factor,
                   opacity_reset_value)

    def change_block_size(self, block_size: int):
        """Change block size of Slang kernels"""

        if block_size != self.block_size:
            self.block_size = block_size
            self.accumulate_norms = self.shader_module.accumulateNorms.call_group_shape(spy.slangpy.Shape(block_size))
            self.count_copies = self.shader_module.countCopies.call_group_shape(spy.slangpy.Shape(block_size))
            self.create_copies = self.shader_module.createCopies.call_group_shape(spy.slangpy.Shape(block_size))

    def accumulate_gradients(self, gradients: torch.Tensor, scales: Sequence[float] | None=None):
        """Add `gradients`' norms to accumulated norms, after applying `scales` to their x and y components"""
        self.accumulate_norms(spy.grid(self.accumulated_norms.shape), self.accumulated_norms, gradients, scales if scales is not None else [1., 1.])

    def adapt_density(self, gs: gauss.Gaussians3D, gs_view_counters: torch.Tensor, gs_img_radii: torch.Tensor, reset_opacity: bool=False):
        """Adapt Gaussian density

        - `gs`: `Gaussians3D` to adapt
        - `gs_view_counters`: Gaussian visibility counters
        - `gs_img_radii`: Gaussian maximum image radii
        - `reset_opacity`: if True, reset Gaussian opacities

        Return a tuple consisting of:
        - kept Gaussian old indices
        - kept Gaussian new indices

        With "kept" meaning Gaussians unaffected by the densification process.
        """

        counts = torch.empty(gs.num, dtype=torch.int32, device=self.accumulated_norms.device)
        cumulative_counts = torch.empty_like(counts)
        are_kept = torch.zeros(gs.num, dtype=torch.bool, device=self.accumulated_norms.device)
        with torch.no_grad():
            self.count_copies(spy.grid((gs.num,)),
                              self.accumulated_norms, gs_view_counters, gs.scales, gs.opacities, gs_img_radii,
                              self.gradient_threshold,
                              gs.use_scale_exponential, self.world_radius_threshold,
                              gs.use_opacity_sigmoid, self.opacity_threshold,
                              self.image_radius_threshold,
                              counts)
            torch.cumsum(counts, dim=0, out=cumulative_counts)

            new_size = cumulative_counts[-1].item()
            new_means = torch.empty((new_size,3), dtype=gs.means.dtype, device=gs.means.device)
            new_rotations = torch.empty((new_size,4), dtype=gs.rotations.dtype, device=gs.rotations.device)
            new_scales = torch.empty((new_size,3), dtype=gs.scales.dtype, device=gs.scales.device)
            new_sh_coefficients = torch.empty((16,new_size,3), dtype=gs.sh_coefficients.dtype, device=gs.sh_coefficients.device)
            new_opacities = torch.empty((new_size,), dtype=gs.opacities.dtype, device=gs.opacities.device)

            self.create_copies(spy.grid((gs.num,)), torch.randint(2**63-1, (), generator=self.prng).item(),
                               counts, cumulative_counts,
                               gs.means, gs.rotations, gs.scales, gs.sh_coefficients, gs.opacities,
                               gs.use_scale_exponential, self.scale_threshold, self.split_factor, reset_opacity, self.opacity_reset_value if not gs.use_opacity_sigmoid else np.log(self.opacity_reset_value / (1. - self.opacity_reset_value)),
                               new_means, new_rotations, new_scales, new_sh_coefficients, new_opacities, are_kept)

        gs.num = new_size
        gs.means = new_means
        gs.rotations = new_rotations
        gs.scales = new_scales
        gs.sh_coefficients = new_sh_coefficients
        gs.opacities = new_opacities
        gs.set_autograd()
        self.accumulated_norms = torch.zeros(new_size, dtype=self.accumulated_norms.dtype, device=self.accumulated_norms.device)

        kept_gaussian_old_indices = torch.nonzero(are_kept)
        kept_gaussian_new_indices = cumulative_counts[kept_gaussian_old_indices] - 1
        return kept_gaussian_old_indices, kept_gaussian_new_indices
