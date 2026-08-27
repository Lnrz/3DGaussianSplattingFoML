from pathlib import Path
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision.io as tv_io
import torchvision.transforms.v2.functional as F
import pycolmap


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


@dataclass
class Camera:
    screensize: Sequence[int]
    intrinsics: torch.Tensor
    half_fov_sin_cos: torch.Tensor
    extrinsics: torch.Tensor


def half_fov_sin_cos(width: int, height: int, focal_length_x: float, focal_length_y: float):
    half_fov_x = np.atan(width / (2. * focal_length_x))
    half_fov_y = np.atan(height / (2. * focal_length_y))
    return np.array([np.sin(half_fov_x), np.cos(half_fov_x), np.sin(half_fov_y), np.cos(half_fov_y)], dtype=np.float32)


class CalibratedImages(Dataset):

    @classmethod
    def from_colmap(cls, path: str, images_dir: str, transform=None):
        dataset = cls()
        rec = pycolmap.Reconstruction(path)
        images = rec.images.values()
        cameras = list(rec.cameras.values())

        dataset.images_dir = images_dir
        dataset.transform = transform
        dataset.image_names = [image.name for image in images]
        dataset.img_to_cam_indices = [cameras.index(image.camera) for image in images]
        dataset.intrinsics = torch.from_numpy(np.array([camera.params for camera in cameras], dtype=np.float32))
        dataset.half_fovs_sin_cos = torch.from_numpy(np.array([half_fov_sin_cos(camera.width, camera.height, camera.focal_length_x, camera.focal_length_y) for camera in cameras], dtype=np.float32))
        dataset.extrinsics = torch.from_numpy(np.array([image.cam_from_world().matrix() for image in images], dtype=np.float32))
        dataset.max_image_size = np.max([[camera.width, camera.height] for camera in cameras], 0)

        camera_centers = np.array([img.projection_center() for img in images])
        world_center = camera_centers.mean(axis=0)
        distances = np.linalg.vector_norm(camera_centers - world_center, axis=1)
        dataset.scene_radius = np.max(distances).item()

        return dataset

    def split_train_test(self):
        group_count = len(self) // 8
        test_indices = [8 * i + 7 for i in range(group_count)]
        test_indices_set = set(test_indices)
        train_indices = [i for i in range(len(self)) if i not in test_indices_set]

        return Subset(self, train_indices), Subset(self, test_indices)
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = Path(self.images_dir) / self.image_names[idx]
        image = tv_io.decode_image(image_path)
        cam_idx = self.img_to_cam_indices[idx]
        intrinsics = self.intrinsics[cam_idx]
        if self.transform:
            image, intrinsics = self.transform(image, intrinsics)

        return image, intrinsics, self.half_fovs_sin_cos[cam_idx], self.extrinsics[idx]

class DownsampleCalibratedImage(torch.nn.Module):

    def __init__(self, factor: int, interpolation: F.InterpolationMode=F.InterpolationMode.BILINEAR, antialias: bool=True):
        super().__init__()
        self.factor = factor
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, image, intrinsics):
        h, w = image.shape[-2:]
        return F.resize(image, [h // self.factor, w // self.factor], self.interpolation, antialias=self.antialias), intrinsics / self.factor
