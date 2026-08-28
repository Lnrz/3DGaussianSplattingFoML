import gc
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure

import splatgs
from splatgs.slang import slang_optim_str_to_enum, slang_fp_mode_str_to_enum


def update_optim_state(optim: Optimizer, gaussians: splatgs.gauss.Gaussians3D, kept_gaussian_old_indices: torch.Tensor, kept_gaussian_new_indices: torch.Tensor):
    new_datas = [
        gaussians.means,
        gaussians.rotations,
        gaussians.scales,
        gaussians.opacities,
        gaussians.sh_coefficients
    ]
    for i, new_data in zip(range(len(optim.param_groups)), new_datas):
        old_data = optim.param_groups[i]["params"][0]
        old_state = optim.state[old_data]

        new_step = old_state["step"]
        new_exp_avg = torch.zeros_like(new_data)
        new_exp_avg_sq = torch.zeros_like(new_data)
        if i != 4:
            new_exp_avg[kept_gaussian_new_indices] = old_state["exp_avg"][kept_gaussian_old_indices]
            new_exp_avg_sq[kept_gaussian_new_indices] = old_state["exp_avg_sq"][kept_gaussian_old_indices]
        else:
            new_exp_avg[:, kept_gaussian_new_indices] = old_state["exp_avg"][:, kept_gaussian_old_indices]
            new_exp_avg_sq[:, kept_gaussian_new_indices] = old_state["exp_avg_sq"][:, kept_gaussian_old_indices]

        optim.param_groups[i]["params"][0] = new_data
        optim.state[new_data] = {
            "step" : new_step,
            "exp_avg" : new_exp_avg,
            "exp_avg_sq" : new_exp_avg_sq
        }
        del optim.state[old_data]


def get_arguments():
    parser = ArgumentParser(description="A script to train Gaussians models")
    parser.add_argument("reconstruction",type=str, help="Path to the COLMAP reconstrution to use as training data.")
    parser.add_argument("-o", "--output", metavar="path", type=str, default="o.ply", help="Path where to save the trained model. Default to 'o.ply'.")
    parser.add_argument("--preset30k", action="store_true", help="Use the 30k iterations preset. The default is the 7k iterations preset.")
    parser.add_argument("--seed", metavar="n", type=int, default=42, help="Seed for the Gaussian splitting PRNG. Default to 42.")
    parser.add_argument("--all-data", action="store_true", help="Add the validation set to the training data. The validation set comprises all images whose index is a multiple of 8.")
    parser.add_argument("--factors", metavar="factor", type=int, nargs="+", default=[1], help="Downscaling factors to apply for training. Default to [1].")
    parser.add_argument("--factors-iters", metavar="iter", type=int, nargs="+", default=[7_000], help="Iterations to reach for each downscaling factor. Default to [7000].")
    parser.add_argument("--densify-from", metavar="iter", type=int, default=500, help="Iteration from which to start the densification process. Default to 500.")
    parser.add_argument("--densify-until", metavar="iter", type=int, default=5_000, help="Iteration at which to stop the densification process. Default to 5000.")
    parser.add_argument("--densification-interval", metavar="n", type=int, default=100, help="Interval at which to densify. Set to 0 to disable. Default to 100.")
    parser.add_argument("--starting-sh-degrees", metavar="deg", type=int, default=0, help="Spherical harmonics degrees trainable at start. Default to 0.")
    parser.add_argument("--sh-degree-interval", metavar="n", type=int, default=1_000, help="Interval at which to increase the spherical harmonics degrees to train. Set to 0 to disable. Default to 1000.")
    parser.add_argument("--opac-reset-interval", metavar="n", type=int, default=3_000, help="Interval at which to reset Gaussian opacities. Set to 0 to disable. Default to 3000.")
    parser.add_argument("--opac-reset-value", metavar="opac", type=float, default=.01, help="The value at which Gaussian opacities are reset to. Must be in range (0-1). Default to 0.01.")
    parser.add_argument("--log-interval", metavar="n", type=int, default=100, help="Interval at which to log training info. Set to 0 to disable. Default to 100.")
    parser.add_argument("--mean-lrs", metavar=("initial", "final"),type=float, nargs=2, default=[1.6e-4, 1.6e-6], help="Initial and final learning rate for Gaussian means."
        " The learning rate of Gaussian means decay exponentially. Default to [1.6e-4, 1.6e-6].")
    parser.add_argument("--rot-lr", metavar="lr", type=float, default=1e-3, help="Learning rate for Gaussian rotations. Default to 1e-3.")
    parser.add_argument("--scale-lr", metavar="lr", type=float, default=5e-3, help="Learning rate for Gaussian scales. Default to 5e-3.")
    parser.add_argument("--opac-lr", metavar="lr", type=float, default=2.5e-2, help="Learning rate for Gaussian opacities. Default to 2.5e-2")
    parser.add_argument("--sh-lr", metavar="lr", type=float, default=2.5e-3, help="Learning rate for Gaussian spherical harmonics coefficients. Default to 2.5e-3.")
    parser.add_argument("--dssim-weight", metavar="weight", type=float, default=.2, help="Weight of dssim loss wrt l1 loss. Default to 0.2.")
    parser.add_argument("--opac-threshold", metavar="opac", type=float, default=.005, help="Opacity below which Gaussians are pruned. Must be in range (0-1). Default to 0.005.")
    parser.add_argument("--gradient-threshold", metavar="thres", type=float, default=6e-6, help="Threshold on the screen position gradient magnitude to determine which Gaussians will be densified. Default to 6e-6.")
    parser.add_argument("--scale-threshold", metavar="thres", type=float, default=.01, help="Threshold on Gaussian scales to determine if a Gaussian will be split or cloned. Relative to the scene extent. Default to 0.01.")
    parser.add_argument("--split-factor", metavar="fact", type=float, default=1.6, help="Factor by which Gaussian scales are divided when splitting. Default to 1.6.")
    parser.add_argument("--scene-radius-perc", metavar="perc", type=float, default=.1, help="Max radius a Gaussian can have relative to the scene extent before being pruned. Default to 0.1.")
    parser.add_argument("--image-radius-perc", metavar="perc", type=float, default=.5, help="Max radius a Gaussian can have relative to the image extent before being pruned. Nonpositive values disable the pruning. Default to 0.5.")
    parser.add_argument("--image-radius-prune-start", metavar="iter", type=int, default=3_001, help="Iteration from which pruning based on image radius will happen. Default to 3001.")
    parser.add_argument("--bg-color", metavar=("r", "g", "b") ,type=float, nargs=3, default=[], help="Background color to use when rendering. Channels must be in range [0-1]. If unset, a random background color will be used at every iteration.")
    parser.add_argument("--optim", choices=["none", "default", "high", "maximal"], default="maximal", help="Optimization level for slang kernel compilation. Default to 'maximal'.")
    parser.add_argument("--fp-mode", choices=["fast", "default", "precise"], default="precise", help="Floating point mode for slang kernel compilation. Default to 'precise'.")
    parser.add_argument("--workers", metavar="n", type=int, default=0, help="Number of workers to use in the training loop. Default to 0.")
    parser.add_argument("--tile-size", metavar="n", type=int, default=16, help="Tile size for rendering. Default to 16.")
    parser.add_argument("--block-size", metavar="n", type=int, default=256, help="Block size for slang kernels other than the rendering kernel."
    " The block size used for rendering is dictated by the value of 'tile-size', not 'block-size'. Default to 256.")
    parser.add_argument("--disable-pin", action="store_true", help="Don't use pinned memory for training data.")
    args = parser.parse_args()

    if args.preset30k:
        args.factors_iters[-1] = 30_000
        args.densify_until = 15_000
    args.densify_from = max(0, args.densify_from)
    args.densification_interval = max(0, args.densification_interval)
    args.starting_sh_degrees = max(0, args.starting_sh_degrees)
    args.opac_reset_interval = max(0, args.opac_reset_interval)
    args.log_interval = max(0, args.log_interval)
    args.dssim_weight = np.clip(args.dssim_weight, .0, 1.)
    args.bg_color = np.clip(args.bg_color, .0, 1.).tolist()
    args.workers = max(-1, args.workers)
    args.optim = slang_optim_str_to_enum(args.optim)
    args.fp_mode = slang_fp_mode_str_to_enum(args.fp_mode)

    if len(args.factors) != len(args.factors_iters):
        parser.error(f"'factors' and 'factors-iters' must have the same length, they were {args.factors} {args.factors_iters}")
    if (np.array(args.factors) <= 0).any():
        parser.error(f"'factors' must be positive, they were {args.factors}")
    if (np.diff(args.factors_iters) <= 0).any():
        parser.error(f"'factors-iters' must be stricly increasing, they were {args.factors_iters}")
    if args.tile_size < 1:
        parser.error(f"'tile-size' must be at least 1, was {args.tile_size}")
    if args.block_size < 1:
        parser.error(f"'block-size' must be at least 1, was {args.block_size}")

    if args.densification_interval != 0:
        if args.densify_until <= args.densify_from:
            parser.error(f"'densify-from' must be less than 'densify-until', they were {args.densify_from} {args.densify_until}")
        if args.opac_reset_value <= .0 or args.opac_reset_value >= 1.:
            parser.error(f"'opac-rest-value' must be in range (0,1), was {args.opac_reset_value}")
        if args.opac_threshold <= .0 or args.opac_threshold >= 1.:
            parser.error(f"'opac-threshold' must be in range (0,1), was {args.opac_threshold}")
        if args.split_factor <= .0:
            parser.error(f"'split-factor' muste be greater than 0, was {args.split_factor}")
        if args.gradient_threshold <= .0:
            parser.error(f"'gradient-threshold' must be greater than 0, was {args.gradient_threshold}")
        if args.scale_threshold <= .0:
            parser.error(f"'scale-threshold' must be greater than 0, was {args.scale_threshold}")
        if args.scene_radius_perc <= .0:
            parser.error(f"'scene-radius-perc' must be greater than 0, was {args.scene_radius_perc}")

    return args


def main():
    args = get_arguments()
    last_iter = args.factors_iters[-1]
    is_log_enabled = args.log_interval != 0
    is_densification_enabled = args.densification_interval != 0
    is_opacity_reset_enabled = args.opac_reset_interval != 0
    is_sh_increase_enabled = args.sh_degree_interval != 0 and args.starting_sh_degrees < 3
    is_image_radius_pruning_enabled = args.image_radius_perc > 0
    is_bg_rand = not args.bg_color
    first_densification_iter = (args.densify_from + args.densification_interval) // args.densification_interval * args.densification_interval if is_densification_enabled else last_iter + 1

    slang_device = splatgs.create_slang_device(optimization_level=args.optim, fp_mode=args.fp_mode)
    render_module = splatgs.load_render_module(slang_device)
    adc_module = splatgs.load_adc_module(slang_device)

    args.reconstruction = Path(args.reconstruction)
    rec_path = str(args.reconstruction / "sparse/0")
    gaussians = splatgs.gs_from_colmap(rec_path, workers=args.workers if (args.workers == -1 or args.workers > 0) else 1, autograd=True)
    ds = splatgs.ds_from_colmap(rec_path, str(args.reconstruction / "images"))
    train_ds, _ = ds.split_train_test()

    ctx = splatgs.ctx(gaussians.num, args.tile_size, args.block_size, render_module, ds.max_image_size)
    backprop_data = splatgs.BackpropagationData.from_settings(gaussians.num, ds.max_image_size)
    densif_data = splatgs.DensificationData.dummy()
    opts = splatgs.RenderOptions(max_sh_degree=args.starting_sh_degrees, background_color=args.bg_color, save_data_for_backprop=True, collect_data_for_densification=first_densification_iter<=0)
    max_width, max_height = ds.max_image_size // min(args.factors)
    out_image = torch.zeros((max_height, max_width, 3), dtype=torch.float32, device="cuda")

    density_control = splatgs.adaptive_density_control(
        gaussians.num,
        adc_module,
        args.block_size,
        args.scale_threshold * ds.scene_radius,
        args.scene_radius_perc * ds.scene_radius,
        args.image_radius_perc * max(max_width, max_height) if is_image_radius_pruning_enabled and (args.image_radius_prune_start == 0) else float("inf"),
        args.opac_threshold,
        args.opac_reset_value,
        args.split_factor,
        args.gradient_threshold,
        args.seed
    )
    mean_lr_start, mean_lr_final = args.mean_lrs
    optim = Adam([
            { "params":gaussians.means, "lr":mean_lr_start * ds.scene_radius },
            { "params":gaussians.rotations, "lr":args.rot_lr },
            { "params":gaussians.scales, "lr":args.scale_lr },
            { "params":gaussians.opacities, "lr":args.opac_lr },
            { "params":gaussians.sh_coefficients, "lr":args.sh_lr }
        ],
        fused=True
    )
    sched = LambdaLR(optim, [lambda iter: (mean_lr_final/mean_lr_start)**(iter/last_iter)] + 4*[lambda iter: 1.])

    curr_iter = 0
    mean_loss = .0
    max_sh_idx = 1
    start_time = time.perf_counter()
    for factor, iters in zip(args.factors, args.factors_iters):
        ds.transform = v2.Compose([splatgs.DownsampleCalibratedImage(factor), v2.ToDtype(torch.float32, scale=True)]) if factor > 1 else v2.ToDtype(torch.float32, scale=True)
        dl = DataLoader(ds if args.all_data else train_ds, shuffle=True, pin_memory=not args.disable_pin, num_workers=args.workers, persistent_workers=args.workers>0)

        if is_log_enabled:
            print(f"[Iter: {curr_iter}] " + (f"Training with {factor}x downsampled images" if factor > 1 else "Traning with images at original resolution"))
        while curr_iter < iters:
            for gt_image, intrinsics, hfsc, extrinsics in dl:
                h, w = gt_image.shape[-2:]
                gt_image: torch.Tensor = gt_image.to(device="cuda", non_blocking=not args.disable_pin)
                intrinsics: torch.Tensor = intrinsics.to(device="cuda", non_blocking=not args.disable_pin)
                hfsc: torch.Tensor = hfsc.to(device="cuda", non_blocking=not args.disable_pin)
                extrinsics: torch.Tensor = extrinsics.to(device="cuda", non_blocking=not args.disable_pin)

                optim.zero_grad(set_to_none=False)
                cam = splatgs.Camera((w,h), intrinsics, hfsc, extrinsics)
                out_image_view = out_image[:h,:w,:]
                if is_bg_rand:
                        opts.background_color = torch.rand(3, device="cuda")
                splatgs.render_gaussians(gaussians, cam, ctx, opts, out_image_view, backprop_data, densif_data)

                out_image_view = out_image_view.permute(2,0,1) # from (H,W,C) to (C,H,W)
                out_image_view = out_image_view.clamp(0.,1.)
                l1 = F.l1_loss(out_image_view, gt_image[0])
                ssim = structural_similarity_index_measure(out_image_view.unsqueeze(0), gt_image, data_range=1.) # parameters need batch dimension (B,C,H,W)
                ldssim = (1. - ssim) / 2.
                loss = l1 * (1. - args.dssim_weight) + ldssim * args.dssim_weight
                loss.backward()
                gaussians.sh_coefficients.grad[1:max_sh_idx].div_(20.) # as in the code of the paper, the lr of the higher order shs is divided by 20.
                optim.step()
                sched.step()

                curr_iter += 1
                loss_cpu = loss.item()
                print(f"[Iter: {curr_iter}] Loss: {loss_cpu:.2e}", end="\r")
                if is_log_enabled:
                        mean_loss += loss_cpu / args.log_interval
                        if curr_iter % args.log_interval == 0:
                            print(f"[Iter: {curr_iter}] Mean Loss: {mean_loss:.2e}")
                            mean_loss = .0

                if curr_iter >= last_iter:
                        break

                if is_sh_increase_enabled and (opts.max_sh_degree < 3) and (curr_iter % args.sh_degree_interval == 0):
                        opts.max_sh_degree += 1
                        max_sh_idx += 2 * opts.max_sh_degree + 1
                        if is_log_enabled:
                            print(f"[Iter: {curr_iter}] Added spherical harmonics degree {opts.max_sh_degree}")

                if is_image_radius_pruning_enabled and (curr_iter == args.image_radius_prune_start):
                        density_control.image_radius_threshold = args.image_radius_perc * max(max_width, max_height)
                        if is_log_enabled:
                            print(f"[Iter: {curr_iter}] Enabled pruning based on Gaussian image radii")

                if is_densification_enabled:
                        if opts.collect_data_for_densification:
                            density_control.accumulate_gradients(ctx.projections.means.grad[:gaussians.num], [w / 2., h / 2.]) # the scales convert the gradient from pixel coordinates to NDC

                            empty_cache = False
                            if curr_iter % args.densification_interval == 0:
                                empty_cache = True
                                old_num = gaussians.num
                                reset_opacity = is_opacity_reset_enabled and ((curr_iter % args.opac_reset_interval) == 0)

                                ketp_gaussian_old_indices, kept_gaussian_new_indices = density_control.adapt_density(gaussians, densif_data.view_counters, densif_data.gaussian_image_radii, reset_opacity)
                                update_optim_state(optim, gaussians, ketp_gaussian_old_indices, kept_gaussian_new_indices)
                                del ketp_gaussian_old_indices, kept_gaussian_new_indices

                                if gaussians.num <= old_num:
                                    densif_data.zero()

                                if is_log_enabled:
                                    print(f"[Iter: {curr_iter}] Gaussians: {gaussians.num:,} ({gaussians.num - old_num:+,})")
                                    if reset_opacity:
                                            print(f"[Iter: {curr_iter}] Opacities were reset")

                            if curr_iter == args.densify_until:
                                empty_cache = True
                                opts.collect_data_for_densification = False
                                densif_data = splatgs.render.DensificationData.dummy()

                            if empty_cache:
                                gc.collect()
                                torch.cuda.empty_cache()

                        if curr_iter == first_densification_iter - args.densification_interval + 1:
                            opts.collect_data_for_densification = True

                if curr_iter >= iters:
                        break

    delta_time = time.perf_counter() - start_time
    gaussians.to_ply(args.output)

    print(f"Gaussian Count: {gaussians.num:,}")
    minutes, seconds = divmod(delta_time, 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Training time: " + (f"{hours:.0f}h " if hours > 0 else "") + f"{minutes:.0f}min {seconds:.3f}s")
    print(f"Saved model to '{args.output}'")


if __name__ == "__main__":
    main()
