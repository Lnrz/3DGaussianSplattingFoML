from collections.abc import Sequence
import argparse
import pathlib
import time
import re

import colorama
import numpy as np
import torch
import taichi as ti

import splatgs
from splatgs.slang import (
    slang_fp_mode_str_to_enum, slang_fp_mode_enum_to_str,
    slang_optim_str_to_enum, slang_optim_enum_to_str
)


colorama.init()
ti.init(arch=ti.cuda)


DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
ERASE_LINE = "\033[K"
ERASE_LINES_ONWARD = "\033[J"


class PovCamera:
    def __init__(self, world_pos: Sequence[float] | None=None, yaw: float=0, pitch: float=0, max_pitch: float=85):
        if world_pos is None:
            world_pos = np.zeros((3,), dtype=np.float64)
        elif not isinstance(world_pos, np.ndarray):
            world_pos = np.array(world_pos, dtype=np.float64)

        self.__world_pos = world_pos
        self.__initial_world_pos = world_pos.copy()
        self.__yaw = np.deg2rad(yaw)
        self.__initial_yaw = self.__yaw.copy()
        self.__pitch = np.deg2rad(np.clip(pitch, -max_pitch, max_pitch))
        self.__initial_pitch = self.__pitch.copy()
        self.__max_pitch = np.deg2rad(max_pitch)

    def get_position(self):
        pos_view = self.__world_pos.view()
        pos_view.flags.writeable = False
        return pos_view

    def get_yaw(self):
        return self.__yaw

    def get_pitch(self):
        return self.__pitch

    def reset(self):
        np.copyto(self.__world_pos, self.__initial_world_pos)
        self.__yaw = self.__initial_yaw
        self.__pitch = self.__initial_pitch

    def rotate_right(self, delta: float, in_radians: bool=True):
        if not in_radians:
            delta = np.deg2rad(delta)
        
        self.__yaw += delta
        if np.abs(self.__yaw) > 2*np.pi:
            self.__yaw -= np.sign(self.__yaw) * 2*np.pi

    def rotate_up(self, delta: float, in_radians: bool=True):
        if not in_radians:
            delta = np.deg2rad(delta)

        self.__pitch += delta
        self.__pitch = np.clip(self.__pitch, -self.__max_pitch, self.__max_pitch)

    # X,Z w.r.t the camera, Y w.r.t the world
    def move(self, delta: Sequence[float]):
        right_dir, _, forward_dir = self.get_camera_frame()
        self.__world_pos += right_dir * delta[0] + forward_dir * delta[2]
        self.__world_pos[1] += delta[1]

    def view_matrix(self, device="cuda"):
        right_dir, up_dir, look_dir = self.get_camera_frame()
        return torch.tensor([
            [right_dir[0], right_dir[1], right_dir[2], -np.dot(right_dir, self.__world_pos)],
            [up_dir[0], up_dir[1], up_dir[2], -np.dot(up_dir, self.__world_pos)],
            [look_dir[0], look_dir[1], look_dir[2], -np.dot(look_dir, self.__world_pos)]
        ], dtype=torch.float32, device=device)

    # Colmap coordinate systems is: X right, Y down, Z forward
    def get_camera_frame(self):
        forward_dir = np.array([np.sin(self.__yaw)*np.cos(self.__pitch), -np.sin(self.__pitch), np.cos(self.__yaw)*np.cos(self.__pitch)], dtype=np.float64)
        right_dir = np.cross([0., 1., 0.], forward_dir)
        right_dir /= np.linalg.vector_norm(right_dir, axis=0, ord=2)
        up_dir = np.cross(forward_dir, right_dir)
        return np.array([right_dir, up_dir, forward_dir])


def resolve_model_path(input_str: str):
    input_path = pathlib.Path(input_str)
    if input_path.exists():
        return input_path

    path_in_data_dir = DATA_DIR / input_path
    if path_in_data_dir.exists():
        return path_in_data_dir
    
    matches = list(DATA_DIR.rglob(input_str))
    if not matches:
        raise FileNotFoundError(f"Found no match in 'data' directory for '{input_str}'")
    elif len(matches) > 2:
        err_msg = f"Found too many matches in 'data' directory for '{input_str}':\n"
        for match in matches:
            err_msg += 4*" " + str(match.relative_to(DATA_DIR)) + "\n"
        raise ValueError(err_msg)

    return matches[0]


def get_arguments():
    parser = argparse.ArgumentParser(description="A script to visualize Gaussian models in ply format.")
    parser.add_argument("model", help="Model path. Can be absolute, relative to the CWD, relative to the 'data' directory, or a search pattern in 'data'.")
    parser.add_argument("--screen-size", type=int, nargs="+", default=[1280,720], metavar="size", help="Window screen size. Pass one value for square windows, two for width and height. Default to 1280x720.")
    parser.add_argument("--initial-position", type=float, nargs=3, default=[0, -1, -2], metavar=("x","y","z"), help="Initial camera position. Default to (0,-1,-2)")
    parser.add_argument("--movement-speed", type=float, default=0.8, metavar="speed", help="Camera movement speed, measured in scene unit. Default to 0.8.")
    parser.add_argument("--rotation-speed", type=float, default=0.8, metavar="speed", help="Camera rotation speed, measured in radians. Default to 0.8.")
    parser.add_argument("--tile-size", type=int, default=16, metavar="size", help="Tile size for rendering. Default to 16.")
    parser.add_argument("--block-size", type=int, default=256, metavar="size", help="Block size for slang kernels other than the rendering kernel."
        " The block size used for rendering is dictated by the value of 'tile_size', not 'block_size'. Default to 256.")
    parser.add_argument("--max-sh-degree", type=int, default=3, metavar="n", help="Maximum spherical harmonics degree for computing color. Default to 3.")
    parser.add_argument("--fovx", type=float, default=60, metavar="fx", help="Camera horizontal field of view, measured in degrees. Default to 60 degrees.")
    parser.add_argument("--near-far", type=float, nargs=2, default=[0.2, 100], metavar=("z_near", "z_far"), help="Near and far plane distances from camera, measured in scene unit. Default to 0.2 100.")
    parser.add_argument("--exponential-resize", action="store_true", help="Enable exponential memory resizing. When capacity is exceeded, memory grows exponentially rather than resizing to the exact size requested.")
    parser.add_argument("--fp-mode", choices=["fast", "default", "precise"], default="fast", help="Floating point mode for slang kernel compilation. Default to fast.")
    parser.add_argument("--optim", choices=["none", "default", "high", "maximal"], default="maximal", help="Optimization level for slang kernel compilation. Default to maximal.")
    args = parser.parse_args()

    try:
        args.model = resolve_model_path(args.model)
    except Exception as e:
        parser.error(e)
    for value in args.screen_size:
        if value <= 0:
            parser.error(f"'screen-size' can't be negative, was {args.screen_size}.")
    if len(args.screen_size) > 2:
        parser.error(f"'screen-size' accepts at most 2 arguments, there were {len(args.screen_size)} arguments.")
    if args.movement_speed <= 0:
        parser.error(f"'movement-speed' can't be less than or equal to 0, was {args.movement_speed}")
    if args.rotation_speed <= 0:
        parser.error(f"'rotation-speed' can't be less than or equal to 0, was {args.rotation_speed}")
    if args.tile_size <= 0:
        parser.error(f"'tile-size' can't be lower than or equal to 0, was {args.tile_size}.")
    if args.block_size <= 0:
            parser.error(f"'block-size' can't be lower than or equal to 0, was {args.block_size}.")
    if args.max_sh_degree < 0 or args.max_sh_degree > 3:
        parser.error(f"'max-sh-degree' can't be negative nor greater than 3, was {args.max_sh_degree}.")
    if args.fovx <= 0 or args.fovx >= 180:
        parser.error(f"'fovx' must be between 0 and 180 extremes excluded, was {args.fovx}.") 
    for distance in args.near_far:
        if distance <= 0:
            parser.error(f"'near-far' can't be negative, was {args.near_far}.")
    if args.near_far[1] <= args.near_far[0]:
        parser.error(f"Near plane can't be further than far plane, was {args.near_far}.")

    args.fp_mode = slang_fp_mode_str_to_enum(args.fp_mode)
    args.optim = slang_optim_str_to_enum(args.optim)

    return args


def get_intrinsics_and_hfsc(width: int, height: int, fov_x: float):
    fov_x = np.deg2rad(fov_x)
    focal_length = width/(2*np.tan(fov_x/2)).item()
    fov_y = 2 *np.atan(height / (2 * focal_length))

    intrinsics = [focal_length, focal_length, width/2, height/2];
    half_fov_xy_sincos = [np.sin(fov_x/2).item(), np.cos(fov_x/2).item(), np.sin(fov_y/2).item(), np.cos(fov_y/2).item()];
    return intrinsics, half_fov_xy_sincos


move_mapping = [
    ("a", (0,-1)), ("d", (0,1)),
    (["q",  ti.ui.SHIFT], (1,1)), (["e", ti.ui.SPACE], (1,-1)),
    ("s", (2,-1)), ("w", (2,1))
]
rotation_mapping = [
    (ti.ui.LEFT, (PovCamera.rotate_right, -1)),
    (ti.ui.RIGHT, (PovCamera.rotate_right, 1)),
    (ti.ui.DOWN, (PovCamera.rotate_up, -1)),
    (ti.ui.UP, (PovCamera.rotate_up, 1))
]


def handle_input(window: ti.ui.Window, povCam: PovCamera, cam: splatgs.Camera, movement_speed: float, rotation_speed: float):
    delta = np.zeros((3,), dtype=np.float64)
    update_view_matrix = False

    if window.is_pressed(ti.ui.ESCAPE):
        window.running = False
        return

    if window.is_pressed("r"):
        povCam.reset()
        update_view_matrix = True

    for key, (dim, sign) in move_mapping:
        if window.is_pressed(*key):
            delta[dim] += sign*movement_speed
            update_view_matrix = True

    for key, (fun, sign) in rotation_mapping:
        if window.is_pressed(key):
            fun(povCam, sign*rotation_speed)
            update_view_matrix = True

    if update_view_matrix:
        povCam.move(delta)
        cam.extrinsics = povCam.view_matrix()


def get_cli_input(message:str, before_input: str=">>> ", types: type | Sequence[type]=str):
    print(message)
    input_string = input(before_input)
    print(colorama.Cursor.UP(2) + ERASE_LINES_ONWARD, end="", flush=True)

    if isinstance(types, type):
        return types(input_string)

    inputs = re.split("[, ]+", input_string)
    if len(inputs) != len(types):
        raise ValueError(f"Too few inputs, were {len(inputs)}, expected {len(types)}")

    return [to_type(input_part) for input_part, to_type in zip(inputs, types)]


def main():
    args = get_arguments()
    model_path = args.model
    movement_speed = args.movement_speed
    rotation_speed = args.rotation_speed
    width = args.screen_size[0]
    height = args.screen_size[1] if len(args.screen_size) > 1 else width
    screen_size = (width, height)
    fov_x = args.fovx

    gaussians = splatgs.gs_from_ply(model_path)
    slang_device = splatgs.create_slang_device(fp_mode=args.fp_mode, optimization_level=args.optim)
    render_module = splatgs.load_render_module(slang_device)
    ctx = splatgs.ctx(gaussians.num, args.tile_size, args.block_size, render_module, screen_size, args.exponential_resize)
    opts = splatgs.RenderOptions(max_sh_degree=args.max_sh_degree, nearFar=args.near_far)
    intrinsics, hfsc = get_intrinsics_and_hfsc(width, height, fov_x)
    povCam = PovCamera(args.initial_position)
    cam = splatgs.Camera(screen_size, intrinsics, hfsc, povCam.view_matrix())
    img = torch.empty((height, width, 3), dtype=torch.float32, device="cuda")

    img_ti = ti.field(ti.f32, (width, height, 3))
    window = ti.ui.Window(name="3D Gaussian Splatting, FoML Project - Visualizer", res=screen_size)
    canvas = window.get_canvas()
    gui = window.get_gui()
    cli_input_notice = "Accept input by command line"
    sep_str = 40*"-"
    prev_time = time.perf_counter()
    while window.running:
        curr_time = time.perf_counter()
        delta_time = np.min([curr_time - prev_time, 0.1])
        prev_time = curr_time

        handle_input(window, povCam, cam, movement_speed*delta_time, rotation_speed*delta_time)
        splatgs.render_gaussians(gaussians, cam, ctx, opts, img)
        img_ti.from_torch(img.clamp_(0., 1.).flip(0).transpose(0,1))
        canvas.set_image(img_ti)

        with gui.sub_window("Position", 0.8, 0, 0.2, 0.1) as pos_gui:
            camera_pos = povCam.get_position()
            pos_gui.text(f"X: {camera_pos[0]:.2f}, Y: {camera_pos[1]:.2f}, Z: {camera_pos[2]:.2f}")
            pos_gui.text(f"Yaw: {np.rad2deg(povCam.get_yaw()):.2f}deg, Pitch: {np.rad2deg(povCam.get_pitch()):.2f}deg")

        with gui.sub_window("Controls", 0.7, 0.7, 0.3, 0.3) as controls_gui:
            controls_gui.text("WASD : move around")
            controls_gui.text("Arrows : look around")
            controls_gui.text("E/Q, Space/Shift : up and down")
            controls_gui.text("R : reset position")
            controls_gui.text(sep_str)
            controls_gui.text(cli_input_notice)
            if controls_gui.button(f"Change movement speed ({movement_speed})"):
                try:
                    movement_speed = get_cli_input("Input movement speed", types=float)
                except Exception as e:
                    print(e)
            if controls_gui.button(f"Change rotation speed ({rotation_speed})"):
                try:
                    rotation_speed = get_cli_input("Input rotation speed", types=float)
                except Exception as e:
                    print(e)
            new_fov = controls_gui.slider_float("Horizontal FOV", fov_x, 30, 150)
            if new_fov != fov_x:
                fov_x = new_fov
                cam.intrinsics, cam.half_fov_sin_cos = get_intrinsics_and_hfsc(width, height, fov_x)

        with gui.sub_window("Info", 0, 0, 0.265, 0.5) as info_gui:
            info_gui.text(f"Model: {model_path.stem}")
            info_gui.text(f"Gaussians in model: {gaussians.num:,d}")
            info_gui.text(f"Gaussian instances: {ctx.instances.num:,d}")
            info_gui.text(f"Memory allocated for {ctx.instances.size:,d} instances")
            info_gui.text(sep_str)
            info_gui.text(f"Image size: ({width},{height})")
            info_gui.text(f"Tile size {ctx.tile_size}, Tiles in image ({ctx.tiles.tiles_xy_cpu[0]},{ctx.tiles.tiles_xy_cpu[1]})")
            info_gui.text(f"Block size {ctx.block_size}")
            info_gui.text(sep_str)
            opts.max_sh_degree = info_gui.slider_int("Max sh degree", opts.max_sh_degree, 0, 3)
            gaussians.use_scale_exponential = info_gui.checkbox("Apply exponential to scale", gaussians.use_scale_exponential)
            gaussians.use_opacity_sigmoid = info_gui.checkbox("Apply sigmoid to opacity", gaussians.use_opacity_sigmoid)
            gaussians.color_bias = info_gui.slider_float("Color bias", gaussians.color_bias, -1, 1)
            info_gui.text(f"Near far planes: ({opts.nearFar[0]},{opts.nearFar[1]})")
            ctx.exponential_resizing = info_gui.checkbox("Exponential memory resize", ctx.exponential_resizing)
            info_gui.text(sep_str)
            info_gui.text(f"Floating point mode: {slang_fp_mode_enum_to_str(args.fp_mode)}")
            info_gui.text(f"Optimization level: {slang_optim_enum_to_str(args.optim)}")
            info_gui.text(sep_str)
            info_gui.text(cli_input_notice)
            if info_gui.button("Change model"):
                new_path = get_cli_input("Input model")
                try:
                    new_path = resolve_model_path(new_path)
                    gaussians = splatgs.gs_from_ply(new_path)
                    model_path = new_path
                except Exception as e:
                    print(e)
            if info_gui.button("Change tile size"):
                try:
                    new_tile_size = get_cli_input("Input tile size", types=int)
                    ctx.change_shader_size(tile_size=new_tile_size)
                except Exception as e:
                    print(e)
            if info_gui.button("Change block size"):
                try:
                    new_block_size = get_cli_input("Input block size", types=int)
                    ctx.change_shader_size(block_size=new_block_size)
                except Exception as e:
                    print(e)
            if info_gui.button("Change near and far planes"):
                try:
                    new_near_far = get_cli_input("Input near far distances", types=[float, float])
                    if new_near_far[0] < new_near_far[1]:
                        opts.nearFar = new_near_far
                    else:
                        print(f"Near plane must be further that far plane, near was {new_near_far[0]}, far was {new_near_far[1]}")
                except Exception as e:
                    print(e)

        window.show()


if __name__ == "__main__":
    main()
