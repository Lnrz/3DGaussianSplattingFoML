import torch
from torch.utils.data import Dataset
import torchvision.io as tv_io
import torchvision.transforms.v2.functional as F
import numpy as np
import pycolmap
from dataclasses import dataclass
from scipy.spatial import KDTree
from plyfile import PlyData
from pathlib import Path


STARTING_OPACITY = -2.19722457734   # logit for 10% initial opacity
N0 = 0.28209479177387814          # normalizing constant of the spherical harmonic Y_0^0


def to_zero_deg_sh_coef(value, scale=1/255, offset=-.5):
    return (value*scale + offset) / N0

def half_fov_sin_cos_from_colmap_camera(camera):
    half_fov_x = np.atan(camera.width / (2. * camera.focal_length_x))
    half_fov_y = np.atan(camera.height / (2. * camera.focal_length_y))
    return np.array([np.sin(half_fov_x), np.cos(half_fov_x), np.sin(half_fov_y), np.cos(half_fov_y)], dtype=np.float32)


@dataclass
class Gaussians3D:
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

    def to_device(self, device):
        self.means = self.means.to(device)
        self.rotations = self.rotations.to(device)
        self.scales = self.scales.to(device)
        self.opacities = self.opacities.to(device)
        self.sh_coefficients = self.sh_coefficients.to(device)

    def set_autograd(self, mode: bool=True):
        self.means.requires_grad_(mode)
        self.rotations.requires_grad_(mode)
        self.scales.requires_grad_(mode)
        self.opacities.requires_grad_(mode)
        self.sh_coefficients.requires_grad_(mode)

@dataclass
class ProjectedGaussians:
    means: torch.Tensor
    depths: torch.Tensor
    covariances: torch.Tensor
    colors: torch.Tensor

    @classmethod
    def from_size(cls, size: int, device="cuda"):
        means = torch.empty([size, 2], dtype=torch.float32, device=device)
        depths = torch.empty(size, dtype=torch.float32, device=device)
        covariances = torch.empty([size, 3], dtype=torch.float32, device=device)
        colors = torch.empty_like(covariances)

        return cls(means, depths, covariances, colors)

    # might leave tensors longer than needed
    def ensure_capacity(self, size: int):
        if size <= self.means.size(0):
            return
        device = self.means.device
        self.means = torch.empty([size, 2], dtype=torch.float32, device=device)
        self.depths = torch.empty(size, dtype=torch.float32, device=device)
        self.covariances = torch.empty([size, 3], dtype=torch.float32, device=device)
        self.colors = torch.empty_like(self.covariances)

    def to_device(self, device):
        self.means = self.means.to(device)
        self.depths = self.depths.to(device)
        self.covariances = self.covariances.to(device)
        self.colors = self.colors.to(device)

@dataclass
class GaussiansInstances:
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
        counts = torch.empty(size, dtype=torch.int32, device=device)
        offsets = torch.empty_like(counts)
        instances = torch.empty(1, dtype=torch.int32, device=device)
        sorted_instances = torch.empty_like(instances)
        keys = torch.empty_like(instances, dtype=torch.int64)
        sorted_keys = torch.empty_like(keys)
        sorted_keys_indices = torch.empty_like(keys)

        return cls(0, 1, counts, offsets, instances, keys, sorted_keys, sorted_keys_indices, sorted_instances)

    # might leave tensors longer than needed
    def ensure_capacity(self, size: int):
        if size <= self.counts.size(0):
            return
        device = self.counts.device
        self.counts = torch.empty(size, dtype=torch.int32, device=device)
        self.cumulative_counts = torch.empty_like(self.counts)

    # might leave tensors longer than needed
    def allocate_instances(self, instances_num:int, by_power_of_two: bool=False):
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
        self.counts = self.counts.to(device)
        self.cumulative_counts = self.cumulative_counts.to(device)
        self.instances = self.instances.to(device)
        self.keys = self.keys.to(device)

@dataclass
class ScreenTiles:
    tiles_xy_cpu: torch.Tensor
    tiles_xy: torch.Tensor
    ranges: torch.Tensor

    @staticmethod
    def screensize_to_tiles(screensize, tile_size: int, device="cpu"):
        tiles_x = (screensize[0] + tile_size - 1) // tile_size
        tiles_y = (screensize[1] + tile_size - 1) // tile_size
        
        return torch.tensor([tiles_x, tiles_y], dtype=torch.int32, device=device), tiles_x * tiles_y
    
    # screensize is (width,height)
    @classmethod
    def from_screensize(cls, screensize, tile_size: int, device="cuda"):
        tiles_xy_cpu, tile_num = cls.screensize_to_tiles(screensize, tile_size, "cpu")
        tiles_xy = tiles_xy_cpu.to(device=device)
        ranges = torch.zeros([tile_num, 2], dtype=torch.int32, device=device)
        
        return cls(tiles_xy_cpu, tiles_xy, ranges)

    # might leave tensors longer than needed
    def ensure_capacity(self, screensize, tile_size: int):
        new_tiles_xy, tile_num = self.screensize_to_tiles(screensize, tile_size, "cpu")

        if not self.tiles_xy_cpu.equal(new_tiles_xy):
            self.tiles_xy_cpu = new_tiles_xy
            self.tiles_xy = new_tiles_xy.to(device=self.tiles_xy.device)
        if self.ranges.size(0) < tile_num:
            self.ranges = torch.zeros([tile_num, 2], dtype=torch.int32, device=self.ranges.device)

class CalibratedImages(Dataset):

    @classmethod
    def from_colmap(cls, path: str, images_dir: str, transform=None):
        dataset = cls()
        rec = pycolmap.Reconstruction(path)

        dataset.images_dir = images_dir
        dataset.transform = transform
        dataset.images_names = [image.name for image in rec.images.values()]
        dataset.intrinsics = torch.from_numpy(np.array([image.camera.params for image in rec.images.values()], dtype=np.float32)) # TODO: there are duplicates here (some datasets have 100s of images but just 1 camera...)
        dataset.half_fovs_sin_cos = torch.from_numpy(np.array([half_fov_sin_cos_from_colmap_camera(image.camera) for image in rec.images.values()], dtype=np.float32)) # TODO: there are duplicates here (some datasets have 100s of images but just 1 camera...)
        dataset.extrinsics = torch.from_numpy(np.array([image.cam_from_world().matrix() for image in rec.images.values()], dtype=np.float32))

        return dataset
    
    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        image_path = Path(self.images_dir) / self.images_names[idx]
        image = tv_io.decode_image(image_path)
        intrinsics = self.intrinsics[idx]
        if self.transform:
            image, intrinsics = self.transform(image, intrinsics)
        
        return image, intrinsics, self.half_fovs_sin_cos[idx], self.extrinsics[idx]
    
class DownsampleCalibratedImage(torch.nn.Module):
    
    def __init__(self, factor: int, interpolation: F.InterpolationMode=F.InterpolationMode.BILINEAR, antialias: bool=True):
        super().__init__()
        self.factor = factor
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, image, intrinsics):
        h, w = image.shape[-2:]
        return F.resize(image, [h // self.factor, w // self.factor], self.interpolation, antialias=self.antialias), intrinsics / self.factor