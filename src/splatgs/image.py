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
    """Class holding screen tile data

    - `tiles_xy`: number of tiles along x and y axes
    - `tiles_xy_cpu`: copy of `tiles_xy` in cpu memory
    - `ranges`: range of Gaussian instances for each tile
    """

    tiles_xy_cpu: torch.Tensor
    tiles_xy: torch.Tensor
    ranges: torch.Tensor

    @staticmethod
    def screensize_to_tiles(screen_size, tile_size: int, device="cpu"):
        """Compute the number of tiles along the x and y axes given `screen_size` and `tile_size`"""

        tiles_x = (screen_size[0] + tile_size - 1) // tile_size
        tiles_y = (screen_size[1] + tile_size - 1) // tile_size

        return torch.tensor([tiles_x, tiles_y], dtype=torch.int32, device=device), tiles_x * tiles_y

    @classmethod
    def from_screen_size(cls, screen_size, tile_size: int, device="cuda"):
        """Create `ScreenTiles` from `screen_size` and `tile_size`

        `screen_size` must contain the width and height of the image in this exact order.
        """

        tiles_xy_cpu, tile_num = cls.screensize_to_tiles(screen_size, tile_size, "cpu")
        tiles_xy = tiles_xy_cpu.to(device=device)
        ranges = torch.zeros([tile_num, 2], dtype=torch.int32, device=device)

        return cls(tiles_xy_cpu, tiles_xy, ranges)

    def ensure_capacity(self, screen_size, tile_size: int):
        """Ensure tensor capacities are big enough for the given `screen_size` and `tile_size`

        If actual capacities are bigger does nothing.
        """

        new_tiles_xy, tile_num = self.screensize_to_tiles(screen_size, tile_size, "cpu")

        if not self.tiles_xy_cpu.equal(new_tiles_xy):
            self.tiles_xy_cpu = new_tiles_xy
            self.tiles_xy = new_tiles_xy.to(device=self.tiles_xy.device)
        if self.ranges.size(0) < tile_num:
            self.ranges = torch.zeros([tile_num, 2], dtype=torch.int32, device=self.ranges.device)


@dataclass
class Camera:
    """Class holding camera parameters

    - `screen_size`: camera width and height in pixels
    - `intrinsics`: camera focal length along x and y axes, and principal point coordinates (both in pixels)
    - `half_fov_sin_cos`: sin(half x fov), cos(half x fov), sin(half y fov), cos(half y fov)
    - `extrinsics`: camera pose as 3x4 matrix [ R | t ]
    """

    screen_size: Sequence[int]
    intrinsics: torch.Tensor
    half_fov_sin_cos: torch.Tensor
    extrinsics: torch.Tensor


def half_fov_sin_cos(width: int, height: int, focal_length_x: float, focal_length_y: float):
    """Given the camera parameters, expressed in pixels, compute the sin and cos of the half fovs

    Return sin(half x fov), cos(half x fov), sin(half y fov), cos(half y fov)
    """

    half_fov_x = np.atan(width / (2. * focal_length_x))
    half_fov_y = np.atan(height / (2. * focal_length_y))
    return np.array([np.sin(half_fov_x), np.cos(half_fov_x), np.sin(half_fov_y), np.cos(half_fov_y)], dtype=np.float32)


class CalibratedImages(Dataset):
    """Dataset of calibrated images

    When iterated over returns:
    - the image
    - its intrinsics
    - sin and cos of its half fovs
    - its extrinsics

    Image is returned as a CxHxW tensor.

    Intrinsics contain:
    - the focal distances (x and y) in pixels
    - the principal point coordinates in pixels

    The sin and cos of half fovs are stored in this order:
    - sin and cos of the half fov along x
    - sin and cos of the half fov along y

    Extrinsics is a 3x4 matrix in the form [ R | t ]
    """

    @classmethod
    def from_colmap(cls, path: str, images_dir: str, transform=None):
        """Create `CalibratedImages` from the COLMAP reconstruction found in `path`, loading images from `images_dir`

        - `transform`: transform to apply to images and intrinsics.
        """

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
        """Return train and test splits of the dataset

        Every 8th image is taken as a test image.
        """

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
    """Transform for downsampling calibrated images"""

    def __init__(self, factor: int, interpolation: F.InterpolationMode=F.InterpolationMode.BILINEAR, antialias: bool=True):
        super().__init__()
        self.factor = factor
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, image, intrinsics):
        h, w = image.shape[-2:]
        return F.resize(image, [h // self.factor, w // self.factor], self.interpolation, antialias=self.antialias), intrinsics / self.factor
