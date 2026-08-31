import argparse
from collections.abc import Sequence

import numpy as np
import torch
from torchvision.transforms import v2
import slangpy as spy
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

import splatgs
from splatgs.image import CalibratedImages
from script_utils import (
    resolve_data_path,
    slang_optim_str_to_enum, slang_fp_mode_str_to_enum
)


class Slider:
    def __init__(self, model_path: str, ds: CalibratedImages, background_color: Sequence[float], optim_lvl: spy.SlangOptimizationLevel, fp_mode: spy.SlangFloatingPointMode, tile_size: int, block_size: int):
        self.curr_idx = 0
        self.ds = ds
        self.gs = splatgs.gs_from_ply(model_path, device="cuda")
        self.opts = splatgs.RenderOptions(background_color=background_color)

        device = splatgs.create_slang_device(optimization_level=optim_lvl, fp_mode=fp_mode)
        module = splatgs.load_render_module(device)
        self.ctx = splatgs.ctx(self.gs.num, tile_size, block_size, module, device="cuda")

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.canvas.manager.set_window_title("Comparison")

        left_img, right_img = self.get_image_pair()
        self.img1 = self.ax1.imshow(left_img)
        self.img2 = self.ax2.imshow(right_img)
        self.ax1.set_title("Gaussian Render")
        self.ax2.set_title("Reconstruction Image")
        self.ax1.axis('off')
        self.ax2.axis('off')

        plt.subplots_adjust(bottom=0.2)
        self.btn_prev = Button(plt.axes([0.3565, 0.12, 0.12, 0.065]), "Previous Image")
        self.btn_next = Button(plt.axes([0.5475, 0.12, 0.1, 0.065]), "Next Image")
        self.btn_prev_valid = Button(plt.axes([0.2865, 0.02, 0.19, 0.065]), "Previous Validation Image")
        self.btn_next_valid = Button(plt.axes([0.5475, 0.02, 0.17, 0.065]), "Next Validation Image")
        self.btn_jump = TextBox(plt.axes([0.065, 0.02, 0.1, 0.065]), "Jump to")
        self.btn_prev.on_clicked(self.prev_pair)
        self.btn_next.on_clicked(self.next_pair)
        self.btn_prev_valid.on_clicked(self.prev_valid_pair)
        self.btn_next_valid.on_clicked(self.next_valid_pair)
        self.btn_jump.on_submit(self.jump_to)
        self.text = plt.figtext(0.01, 0.95, f"{self.curr_idx + 1}/{len(self.ds)}")

    def get_image_pair(self):
        gt_img, intrinsics, hfsc, extrinsics = self.ds[self.curr_idx]
        height, width = gt_img.shape[-2:]

        cam = splatgs.Camera([width, height], intrinsics.to(device="cuda"), hfsc.to(device="cuda"), extrinsics.to(device="cuda"))
        pred_img = splatgs.render_gaussians(self.gs, cam, self.ctx, self.opts)[0]

        gt_img = gt_img.permute(1,2,0).numpy(force=True)
        pred_img = pred_img.clamp_(0.,1.).numpy(force=True)

        return pred_img, gt_img

    def update(self):
        left_img, right_img = self.get_image_pair()

        self.img1.set_data(left_img)
        self.img2.set_data(right_img)
        self.text.set_text(f"{self.curr_idx + 1}/{len(self.ds)}")
        self.fig.canvas.draw_idle()

    def next_pair(self, event):
        self.curr_idx = (self.curr_idx + 1) % len(self.ds)
        self.update()

    def prev_pair(self, event):
        self.curr_idx = (self.curr_idx - 1) % len(self.ds)
        self.update()

    def next_valid_pair(self, event):
        self.curr_idx = ((self.curr_idx + 1 + 8) // 8 * 8) - 1
        if self.curr_idx > len(self.ds):
            self.curr_idx = 7
        self.update()

    def prev_valid_pair(self, event):
        self.curr_idx = self.curr_idx // 8 * 8 - 1
        if self.curr_idx <= 0:
            self.curr_idx = len(self.ds) // 8 * 8 - 1
        self.update()

    def jump_to(self, str):
        if not str:
            return

        self.btn_jump.set_val("")
        try:
            self.curr_idx = min(max(int(str), 1), len(self.ds)) - 1
            self.update()
        except:
            print(f"Invalid index '{str}'")


def get_args():
    parser = argparse.ArgumentParser(description="Script to compare Gaussian model renders with reconstruction images.")
    parser.add_argument("reconstruction", type=str, help="Path to COLMAP reconstrution. Can be absolute, relative to the CWD, relative to the 'data' directory, or a search pattern in 'data'.")
    parser.add_argument("model", type=str, help="Path to PLY Gaussian model.")
    parser.add_argument("--downscale-factor", metavar="factor", type=int, default=1, help="Downscaling factor to use. Default to 1.")
    parser.add_argument("--background-color", metavar=("r", "g", "b"), nargs=3, type=float, default=[.5, .0, 1.], help="Background color to use. Default to purple.")
    parser.add_argument("--optim", choices=["none", "default", "high", "maximal"], default="maximal", help="Optimization level for slang kernel compilation. Default to 'maximal'.")
    parser.add_argument("--fp-mode", choices=["fast", "default", "precise"], default="fast", help="Floating point mode for slang kernel compilation. Default to 'fast'.")
    parser.add_argument("--tile-size", metavar="n", type=int, default=16, help="Tile size for rendering. Default to 16.")
    parser.add_argument("--block-size", metavar="n", type=int, default=256, help="Block size for slang kernels other than the rendering kernel."
    " The block size used for rendering is dictated by the value of 'tile-size', not 'block-size'. Default to 256.")

    args = parser.parse_args()

    args.reconstruction = str(resolve_data_path(args.reconstruction))
    args.model = str(resolve_data_path(args.model))
    args.optim = slang_optim_str_to_enum(args.optim)
    args.fp_mode = slang_fp_mode_str_to_enum(args.fp_mode)
    args.background_color = np.clip(args.background_color, 0., 1.).tolist()
    if args.downscale_factor < 1:
        parser.error(f"'downscale-factor' must be at least 1, was {args.downscale_factor}")
    if args.tile_size < 1:
        parser.error(f"'tile-size' must be at least 1, was {args.tile_size}")
    if args.block_size < 1:
        parser.error(f"'block-size' must be at least 1, was {args.block_size}")

    return args


def main():
    args = get_args()
    ds = splatgs.ds_from_colmap(
        args.reconstruction + R"\sparse\0",
        args.reconstruction + R"\images",
        v2.Compose([splatgs.DownsampleCalibratedImage(args.downscale_factor), v2.ToDtype(torch.float32, True)]) if args.downscale_factor > 1 else v2.ToDtype(torch.float32, True)
    )
    slider = Slider(args.model, ds, args.background_color, args.optim, args.fp_mode, args.tile_size, args.block_size)
    plt.show()


if __name__ == "__main__":
    main()