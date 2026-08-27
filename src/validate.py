from typing_extensions import Self
from pathlib import Path
import argparse
import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure, learned_perceptual_image_patch_similarity
import slangpy as spy

import data
import render


class ValidationNode:

    @classmethod
    def root(cls, reconstructions_path: str | Path, models_path: str | Path):
        reconstructions_path = reconstructions_path if isinstance(reconstructions_path, Path) else Path(reconstructions_path)
        models_path = models_path if isinstance(models_path, Path) else Path(models_path)
        root = cls()
        root.parent = None
        root.children = []
        root.path_part = reconstructions_path.name
        root.reconstruction_path_part = reconstructions_path
        root.model_path_part = models_path
        root.image_count = -1
        root.psnr = -1.
        root.ssim = -1.
        root.lpips = -1.

        if not ValidationNode.is_colmap_folder(reconstructions_path):
            for child_path in reconstructions_path.iterdir():
                if child_path.is_file():
                    continue
                root.children.append(ValidationNode._child(root, child_path, models_path / child_path.name))

        return root

    @classmethod
    def _child(cls, parent: Self, reconstructions_path: Path, models_path: Path):
        node = cls()
        node.parent = parent
        node.children = []
        node.path_part = reconstructions_path.name
        node.image_count = -1
        node.psnr = -1.
        node.ssim = -1.
        node.lpips = -1.

        if not node.is_colmap_folder(reconstructions_path):
            for child_path in reconstructions_path.iterdir():
                if child_path.is_file():
                    continue
                node.children.append(ValidationNode._child(node, child_path, models_path / child_path.name))

        return node

    @staticmethod
    def is_colmap_folder(path: Path):
        return (path / "images").exists() and (path / "sparse/0").exists()

    def print_validation(self, level: int=0, file=None):
        print(level * "    " +  f"{self.path_part}   [PSNR {self.psnr:.2f}   SSIM {self.ssim:.4f}   LPIPS {self.lpips:.4f}   IMAGE COUNT {self.image_count}]", file=file)

        for child in self.children:
            child.print_validation(level + 1, file)

        if self.children:
            print(file=file)

    def get_full_paths(self):
        if not self.parent:
            return (self.reconstruction_path_part, self.model_path_part)

        parent_paths = self.parent.get_full_paths()
        return (parent_paths[0] / self.path_part, parent_paths[1] / self.path_part)

    def validate(self, ctx: render.RenderContext, opts: dict={}, workers: int=0, pin_memory: bool=True):
        self.factor = opts.get(self.path_part, {}).get("factor", 1 if self.parent is None else self.parent.factor)
        self.background = opts.get(self.path_part, {}).get("background", [.0, .0, .0] if self.parent is None else self.parent.background)

        for child in self.children:
            child.validate(ctx, opts, workers, pin_memory)

        if self.children:
            self.image_count = sum([child.image_count for child in self.children])
            self.psnr = sum([child.psnr_sum for child in self.children]) / self.image_count
            self.ssim = sum([child.ssim_sum for child in self.children]) / self.image_count
            self.lpips = sum([child.lpips_sum for child in self.children]) / self.image_count
        else:
            reconstruction_path, model_path = self.get_full_paths()
            model_path = model_path.with_suffix(".ply")

            gaussians = data.Gaussians3D.from_ply(model_path)
            ds = data.CalibratedImages.from_colmap(
                reconstruction_path / "sparse/0",
                reconstruction_path / "images",
                v2.ToDtype(torch.float32, True) if self.factor == 1 else v2.Compose([data.DownsampleCalibratedImage(self.factor), v2.ToDtype(torch.float32, scale=True)])
            )
            _, ds_test = ds.split_train_test()
            dl = DataLoader(ds_test, num_workers=workers, pin_memory=pin_memory)

            psnrs = []
            ssims = []
            lpipss = []
            render_opts = render.RenderOptions(background_color=self.background)
            for i, (gt_img, intr, hfsc, extr) in enumerate(dl):
                print(f"Processing image {i+1}/{len(dl)} of {reconstruction_path}", end="\r")
                gt_img = gt_img.to(device="cuda", non_blocking=pin_memory)
                intr = intr.to(device="cuda", non_blocking=pin_memory)
                hfsc = hfsc.to(device="cuda", non_blocking=pin_memory)
                extr = extr.to(device="cuda", non_blocking=pin_memory)
                h, w = gt_img.shape[-2:]
                cam = render.Camera((w,h), intr, hfsc, extr)

                pred_img = render.render(gaussians, cam, ctx, render_opts)[0]
                pred_img.clamp_(0., 1.)
                pred_img = pred_img.permute(2, 0, 1)
                pred_img = pred_img.unsqueeze_(0)

                psnrs.append(peak_signal_noise_ratio(pred_img, gt_img, data_range=1.))
                ssims.append(structural_similarity_index_measure(pred_img, gt_img, data_range=1.))
                lpipss.append(learned_perceptual_image_patch_similarity(pred_img, gt_img, net_type="alex", normalize=True))
            print()

            psnrs = torch.stack(psnrs).numpy(force=True)
            ssims = torch.stack(ssims).numpy(force=True)
            lpipss = torch.stack(lpipss).numpy(force=True)
            self.psnr = psnrs.mean()
            self.ssim = ssims.mean()
            self.lpips = lpipss.mean()
            self.image_count = len(dl)
            self.psnr_sum = psnrs.sum()
            self.ssim_sum = ssims.sum()
            self.lpips_sum = lpipss.sum()


class DictPairAction(argparse.Action):

    def __init__(self, key_type=str, value_type=str, *args, **kwargs):
        self.key_type = key_type
        self.value_type = value_type
        super(DictPairAction, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if not hasattr(namespace, self.dest):
            setattr(namespace, self.dest, {})

        res_dict = getattr(namespace, self.dest)

        if not isinstance(values, list):
            values = [values]

        for string in values:
            try:
                key, val = re.split("[=:]", string, maxsplit=1)
                res_dict[self.key_type(key)] = self.value_type(val)
            except Exception as e:
                parser.error(f"argument --{self.dest}: pair '{string}' is ill-formed. Correct format is 'key=value' or 'key:value'.\n{e}")


slang_optim_str_to_enum = {
    "none" : spy.SlangOptimizationLevel.none,
    "default" : spy.SlangOptimizationLevel.default,
    "high" : spy.SlangOptimizationLevel.high,
    "maximal" : spy.SlangOptimizationLevel.maximal
}

slang_fp_mode_str_to_enum = {
    "fast" : spy.SlangFloatingPointMode.fast,
    "default" : spy.SlangFloatingPointMode.default,
    "precise" : spy.SlangFloatingPointMode.precise
}

def str_to_color(string: str):
    color = re.split(",", string, maxsplit=2)
    if len(color) != 3:
        raise ValueError(f"Input must have format 'r,g,b', was '{string}'")

    color = [float(channel) for channel in color]

    return color

def get_args():
    parser = argparse.ArgumentParser(description="A script to validate Gaussian models")
    parser.add_argument("datasets", type=str, help="Path to the base directory containing all the validation datasets.")
    parser.add_argument("models", type=str, help="Path to the base directory containing all the models to validate.")
    parser.add_argument("-o", "--output", metavar="path", type=str, default="", help="Path where to save the metrics. The metrics will be printed to console if not specified.")
    parser.add_argument("--factors", metavar="dataset=factor", action=DictPairAction, value_type=int, nargs="+", default={}, help="Downscaling factors to apply to the datasets.")
    parser.add_argument("--backgrounds", metavar="dataset=r,g,b", action=DictPairAction, value_type=str_to_color, nargs="+", default={}, help="Background colors to use for rendering.")
    parser.add_argument("--workers", metavar="n", type=int, default=0, help="Number of workers to use in the validation loop. Default to 0.")
    parser.add_argument("--disable-pin", action="store_true", help="Don't use pinned memory for validation data.")
    parser.add_argument("--optim", choices=["none", "default", "high", "maximal"], default="maximal", help="Optimization level for slang kernel compilation. Default to 'maximal'.")
    parser.add_argument("--fp-mode", choices=["fast", "default", "precise"], default="fast", help="Floating point mode for slang kernel compilation. Default to 'fast'.")
    parser.add_argument("--tile-size", metavar="n", type=int, default=16, help="Tile size for rendering. Default to 16.")
    parser.add_argument("--block-size", metavar="n", type=int, default=256, help="Block size for slang kernels other than the rendering kernel."
    " The block size used for rendering is dictated by the value of 'tile-size', not 'block-size'. Default to 256.")
    args = parser.parse_args()

    for factor in args.factors.values():
        if factor < 1:
            parser.error(f"Downscaling factors must be at least 1, there was {factor}")
    for color in args.backgrounds.values():
        color = np.array(color)
        if (color < 0).any() or (color > 1).any():
            parser.error(f"Background colors must have their channels in range [0-1], there was {color}")
    if args.tile_size < 1:
        parser.error(f"'tile-size' must be at least 1, was {args.tile_size}")
    if args.block_size < 1:
        parser.error(f"'block-size' must be at least 1, was {args.block_size}")

    args.workers = max(-1, args.workers)
    args.optim = slang_optim_str_to_enum[args.optim]
    args.fp_mode = slang_fp_mode_str_to_enum[args.fp_mode]
    args.opts = {}
    for dataset, factor in args.factors.items():
        if args.opts.get(dataset) is None:
            args.opts[dataset] = {}

        args.opts[dataset]["factor"] = factor
    for dataset, background in args.backgrounds.items():
        if args.opts.get(dataset) is None:
            args.opts[dataset] = {}
    
        args.opts[dataset]["background"] = background

    return args


def main():
    args = get_args()
    slang_device = render.create_slangpy_device_for_torch(fp_mode=args.fp_mode, optimization_level=args.optim, include_paths=[str(Path(__file__).parent)])
    render_module = spy.Module.load_from_file(slang_device, "render.slang")
    ctx = render.RenderContext.from_settings(1, args.tile_size, args.block_size, render_module)

    root = ValidationNode.root(args.datasets, args.models)
    root.validate(ctx, args.opts, args.workers, not args.disable_pin)

    if args.output:
        with open(args.output, "a") as f:
            root.print_validation(file=f)
        print(f"Saved results to {args.output}")
    else:
        root.print_validation()


if __name__ == "__main__":
    main()
