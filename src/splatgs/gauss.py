from pathlib import Path
from dataclasses import dataclass

import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
from scipy.spatial import KDTree
import torch
import pycolmap
from plyfile import PlyData, PlyElement


STARTING_OPACITY = -2.19722457734
"""Starting Gaussian opacity (logit for a 10% initial opacity)"""
N0 = 0.28209479177387814
"""Normalizing constant of the spherical harmonic Y_0^0"""


def to_zero_deg_sh_coef(value, scale=1/255, offset=-.5):
    """Convert `value` to zero degree spherical harmonics coefficient, after applying `scale` and `offset`"""
    return (value * scale + offset) / N0


@dataclass
class Gaussians3D:
    """Class holding Gaussian model data

    - `num`: number of Gaussians in the model
    - `means`: Gaussian means
    - `rotations`: Gaussian rotations, stored as quaternions (w,x,y,z)
    - `scales`: Gaussian scales
    - `opacities`: Gaussian opacities
    - `sh_coefficients`: Gaussian spherical harmonics coefficients, shape (16,Gaussians,rgb)
    - `use_opacity_sigmoid`: apply sigmoid to Gaussian opacities when rendering
    - `use_scale_exponential`: apply exponential to Gaussian scales when rendering
    - `color_bias`: bias to apply to Gaussian RGB colors after conversion from SH coefficients
    """

    num: int

    means: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    sh_coefficients: torch.Tensor

    use_opacity_sigmoid: bool
    use_scale_exponential: bool
    color_bias: float

    @classmethod
    def from_colmap(cls, path: str, workers: int=1, device="cuda", autograd: bool=False):
        """Convert COLMAP reconstruction into Gaussian model

        - `path`: COLMAP reconstruction path
        - `workers`: number of threads to use when calculating Gaussian scales
        - `device`: Torch device into which to store Gaussian data
        - `autograd`: enable gradient tracking for Gaussian data
        """

        rec = pycolmap.Reconstruction(path)
        num_points = rec.num_points3D()

        means = torch.from_numpy(np.array([point.xyz for point in rec.points3D.values()], dtype=np.float32))
        # the scale is set to the mean distance of the three closest points
        scales = torch.from_numpy(
            np.log(
                np.mean(
                    KDTree(means).query(means, 4, workers=workers)[0][:,1:]
                    , axis=1, keepdims=True, dtype=np.float32))
            ).to(device=device).repeat((1,3)).requires_grad_(autograd)
        means = means.to(device=device).requires_grad_(autograd)
        rotations = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device).repeat((num_points, 1)).requires_grad_(autograd)
        opacities = torch.tensor([STARTING_OPACITY], dtype=torch.float32, device=device).repeat((num_points)).requires_grad_(autograd)
        sh_coefficients = torch.from_numpy(
            np.pad(
                np.expand_dims(
                    np.array([
                        [to_zero_deg_sh_coef(point.color[0]), to_zero_deg_sh_coef(point.color[1]), to_zero_deg_sh_coef(point.color[2])]
                        for point in rec.points3D.values()], dtype=np.float32)
                    , 0)
                , [(0,15),(0,0),(0,0)])
        ).to(device=device).requires_grad_(autograd)

        return cls(num_points, means, rotations, scales, opacities, sh_coefficients, True, True, 0.5)

    @classmethod
    def from_ply(cls, path: str, use_opacity_sigmoid: bool=True, use_scale_exponential: bool=True, color_bias: float=0.5, device="cuda", autograd: bool=False):
        """Load Gaussian model from PLY file

        - `path`: PLY file path
        - `use_opacity_sigmoid`: apply sigmoid to Gaussian opacities when rendering
        - `use_scale_exponential`: apply exponential to Gaussian scales when rendering
        - `color_bias`: bias to apply to Gaussian RGB colors
        - `device`: Torch device into which to store Gaussian data
        - `autograd`: enable gradient tracking for Gaussian data

        The function expects that:
        - means are stored in the "x", "y", and "z" properties
        - rotations in the "rot_[0-3]" properties, in order w, x, y, z
        - scales in the "scale_[0-2]" properties
        - opacities in the "opacity" property
        - SH coefficients in the "f_dc_[0-2]" and "f_rest_[0-44]" properties

        The function also expects that "f_rest_[0-44]" store channel coefficients contiguously, that is, first all the red coefficients, then all the green coefficients, and finally all the blue coefficients.
        """

        model = PlyData.read(path)
        vertices = model["vertex"]

        num = vertices.count
        means = torch.from_numpy(np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)).to(device=device).requires_grad_(autograd)
        rotations = torch.from_numpy(np.column_stack((vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"])).astype(np.float32)).to(device=device).requires_grad_(autograd)
        scales = torch.from_numpy(np.column_stack((vertices["scale_0"], vertices["scale_1"], vertices["scale_2"])).astype(np.float32)).to(device=device).requires_grad_(autograd)
        opacities = torch.from_numpy(vertices["opacity"].astype(np.float32)).to(device=device).requires_grad_(autograd)
        dc_properties = [f"f_dc_{i}" for i in range(3)]
        rest_properties = [f"f_rest_{i}" for i in range(45)]
        dc = np.expand_dims(np.stack([vertices[prop] for prop in dc_properties], dtype=np.float32).T, 0)                        # (1(dc),gaussians,rgb)
        rest = np.stack([vertices[prop] for prop in rest_properties], dtype=np.float32).T.reshape(-1, 3, 15).transpose(2, 0, 1) # (15(rest),gaussians,rgb)
        sh_coefficients = torch.from_numpy(np.concatenate([dc, rest], axis=0)).to(device=device).requires_grad_(autograd)       # (sh_coefficients,gaussians,rgb)

        return cls(num, means, rotations, scales, opacities, sh_coefficients, use_opacity_sigmoid, use_scale_exponential, color_bias)

    def to_ply(self, path: str | Path):
        """Save Gaussian model as PLY file in `path`

        - means are stored in the "x", "y", and "z" properties
        - rotations in the "rot_[0-3]" properties, in order w, x, y, z
        - scales in the "scale_[0-2]" properties
        - opacities in the "opacity" property
        - SH coefficients in the "f_dc_[0-2]" and "f_rest_[0-44]" properties

        For the "f_rest_[0-44]" properties, the SH coefficients are saved contiguously per channel, that is, first all the red coefficients, then all the green coefficients, and finally all the blue coefficients.
        """

        means = self.means.numpy(force=True)
        rotations = self.rotations.numpy(force=True)
        scales = self.scales.numpy(force=True)
        opacities = self.opacities.numpy(force=True).reshape((self.num, 1))
        dcs = self.sh_coefficients[0].numpy(force=True)
        rests = np.permute_dims(self.sh_coefficients[1:].numpy(force=True), [1, 2, 0]).reshape((self.num, -1))
        vertex_data = np.concat([means, rotations, scales, opacities, dcs, rests], axis=1)
        vertex_data = unstructured_to_structured(vertex_data,
            dtype= [("x","f4"), ("y","f4"), ("z","f4")] +
                   [(f"rot_{i}","f4") for i in range(4)] +
                   [(f"scale_{i}","f4") for i in range(3)] +
                   [("opacity","f4")] +
                   [(f"f_dc_{i}","f4") for i in range(3)] +
                   [(f"f_rest_{i}","f4") for i in range(45)]
        )

        PlyData([PlyElement.describe(vertex_data, "vertex")]).write(path)

    def to_device(self, device):
        """Move data to `device`"""
        self.means = self.means.to(device)
        self.rotations = self.rotations.to(device)
        self.scales = self.scales.to(device)
        self.opacities = self.opacities.to(device)
        self.sh_coefficients = self.sh_coefficients.to(device)

    def set_autograd(self, mode: bool=True):
        """Set gradient tracking"""
        self.means.requires_grad_(mode)
        self.rotations.requires_grad_(mode)
        self.scales.requires_grad_(mode)
        self.opacities.requires_grad_(mode)
        self.sh_coefficients.requires_grad_(mode)


@dataclass
class ProjectedGaussians:
    """Class holding projected Gaussian data

    - `means`: Gaussian means in screen space
    - `depths`: Gaussian depths in camera coordinates
    - `covariances`: Gaussian covariance matrices, stored as xvar, yvar, cov
    - `colors`: Gaussian RGB colors
    """

    means: torch.Tensor
    depths: torch.Tensor
    covariances: torch.Tensor
    colors: torch.Tensor

    @classmethod
    def from_size(cls, size: int, device="cuda"):
        """Initialize `ProjectedGaussian` tensors for `size` Gaussians"""

        means = torch.empty([size, 2], dtype=torch.float32, device=device)
        depths = torch.empty(size, dtype=torch.float32, device=device)
        covariances = torch.empty([size, 3], dtype=torch.float32, device=device)
        colors = torch.empty_like(covariances)

        return cls(means, depths, covariances, colors)

    def ensure_capacity(self, size: int):
        """Ensure tensor capacities are big enough for `size` Gaussians

        If actual capacities are bigger does nothing.
        """

        if size <= self.means.size(0):
            return
        device = self.means.device
        self.means = torch.empty([size, 2], dtype=torch.float32, device=device)
        self.depths = torch.empty(size, dtype=torch.float32, device=device)
        self.covariances = torch.empty([size, 3], dtype=torch.float32, device=device)
        self.colors = torch.empty_like(self.covariances)

    def to_device(self, device):
        """Move data to `device`"""
        self.means = self.means.to(device)
        self.depths = self.depths.to(device)
        self.covariances = self.covariances.to(device)
        self.colors = self.colors.to(device)


@dataclass
class GaussiansInstances:
    """Class holding Gaussian instance data

    - `num`: number of instances
    - `size`: maximum number of instances given current tensor sizes
    - `counts`: number of tile-Gaussian intersections per Gaussian
    - `cumulative_counts`: cumulative sum of `counts`
    - `instances`: Gaussian instances (i.e. indices to the corresponding Gaussians)
    - `keys`: instance keys
    - `sorted_keys`: sorted `keys`
    - `sorted_keys_indices`: sorting indices of `keys`
    - `sorted_instances`: sorted `instances`
    """

    num: int
    size: int

    counts: torch.Tensor
    cumulative_counts: torch.Tensor

    instances: torch.Tensor
    keys: torch.Tensor

    sorted_keys: torch.Tensor
    sorted_keys_indices: torch.Tensor
    sorted_instances: torch.Tensor

    @classmethod
    def from_size(cls, size: int, device="cuda"):
        """Initialize `ProjectedGaussian` tensors for `size` Gaussians"""

        counts = torch.empty(size, dtype=torch.int32, device=device)
        offsets = torch.empty_like(counts)
        instances = torch.empty(1, dtype=torch.int32, device=device)
        sorted_instances = torch.empty_like(instances)
        keys = torch.empty_like(instances, dtype=torch.int64)
        sorted_keys = torch.empty_like(keys)
        sorted_keys_indices = torch.empty_like(keys)

        return cls(0, 1, counts, offsets, instances, keys, sorted_keys, sorted_keys_indices, sorted_instances)

    def ensure_capacity(self, size: int):
        """Ensure tensor capacities are big enough for `size` Gaussians

        If actual capacities are bigger does nothing.
        """

        if size <= self.counts.size(0):
            return
        device = self.counts.device
        self.counts = torch.empty(size, dtype=torch.int32, device=device)
        self.cumulative_counts = torch.empty_like(self.counts)

    def allocate_instances(self, instances_num:int, by_power_of_two: bool=False):
        """Ensure tensor capacities are big enough for `instaces_num` instances

        If actual capacities are bigger does nothing.

        If `by_power_of_two` is True, the new capacities will be the next power of two from `instances_num`.
        """

        self.num = instances_num
        if (self.size < self.num):
            if by_power_of_two:
                two_exp = np.ceil(np.log2(self.num/self.size)).item()
                self.size = int(self.size * (2 ** two_exp))
            else:
                self.size = self.num
            self.instances = torch.empty(self.size, dtype=torch.int32, device=self.counts.device)
            self.sorted_instances = torch.empty_like(self.instances)
            self.keys = torch.empty_like(self.instances, dtype=torch.int64)
            self.sorted_keys = torch.empty_like(self.keys)
            self.sorted_keys_indices = torch.empty_like(self.keys)

    def to_device(self, device):
        """Move data to `device`"""
        self.counts = self.counts.to(device)
        self.cumulative_counts = self.cumulative_counts.to(device)
        self.instances = self.instances.to(device)
        self.keys = self.keys.to(device)
