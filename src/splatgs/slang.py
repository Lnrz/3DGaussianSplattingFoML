import pathlib

import torch
import slangpy as spy


SPLATGS_SHADER_PATH = str(pathlib.Path(__file__).parent / "shaders")
SPLATGS_RENDER_SHADER = "render.slang"
SPLATGS_ADC_SHADER = "adc.slang"

_slang_fp_mode_str_to_enum_dict = {
    "fast" : spy.SlangFloatingPointMode.fast,
    "default" : spy.SlangFloatingPointMode.default,
    "precise" : spy.SlangFloatingPointMode.precise
}
_slang_fp_mode_enum_to_str_dict = {enum : string for string, enum in _slang_fp_mode_str_to_enum_dict.items()}

_slang_optim_str_to_enum_dict = {
    "none" : spy.SlangOptimizationLevel.none,
    "default" : spy.SlangOptimizationLevel.default,
    "high" : spy.SlangOptimizationLevel.high,
    "maximal" : spy.SlangOptimizationLevel.maximal
}
_slang_optim_enum_to_str_dict = {enum : string for string, enum in _slang_optim_str_to_enum_dict.items()}


def slang_fp_mode_str_to_enum(fp_mode: str):
    res = _slang_fp_mode_str_to_enum_dict.get(fp_mode.lower())
    if res is None:
        valid_modes = ", ".join(f"'{k}'" for k in _slang_fp_mode_str_to_enum_dict.keys())
        raise ValueError(
            f"'{fp_mode}' is not a valid floating point mode.\n" +
            "Valid modes are:" + valid_modes + "."
        )

    return res


def slang_fp_mode_enum_to_str(fp_mode: spy.SlangFloatingPointMode):
    return _slang_fp_mode_enum_to_str_dict[fp_mode]


def slang_optim_str_to_enum(optim: str):
    res = _slang_optim_str_to_enum_dict.get(optim.lower())
    if res is None:
        valid_levels = ", ".join(f"'{k}'" for k in _slang_optim_str_to_enum_dict.keys())
        raise ValueError(
            f"'{optim}' is not a valid optimization level.\n" +
            "Valid levels are:" + valid_levels + "."
        )

    return res


def slang_optim_enum_to_str(optim: spy.SlangOptimizationLevel):
    return _slang_optim_enum_to_str_dict[optim]


def create_slang_device(
        type: spy.DeviceType=spy.DeviceType.cuda,
        include_paths=[],
        torch_device=None,
        fp_mode: spy.SlangFloatingPointMode=spy.SlangFloatingPointMode.default,
        optimization_level: spy.SlangOptimizationLevel=spy.SlangOptimizationLevel.default,
        debug_info: spy.SlangDebugInfoLevel = spy.SlangDebugInfoLevel.standard,
        enable_debug_layers: bool=False,
        enable_print: bool=False):
    torch.cuda.init()
    torch.cuda.current_device()
    torch.cuda.current_stream()
    if torch_device is None:
        torch_device = torch.cuda.current_device()
    with torch.device(torch_device):
        handles = spy.get_cuda_current_context_native_handles()

    return spy.Device(
        type=type,
        compiler_options= {
            "include_paths": [spy.SHADER_PATH, SPLATGS_SHADER_PATH] + include_paths,
            "floating_point_mode" : fp_mode,
            "optimization" : optimization_level,
            "disable_warnings" : ["31000"], # warning about shared memory array of link-time constant length being experimental
            "debug_info" : debug_info,
        },
        enable_debug_layers=enable_debug_layers,
        enable_print=enable_print,
        enable_cuda_interop=(type != spy.DeviceType.cuda),
        existing_device_handles=handles,
    )


def load_render_module(device: spy.Device):
    return spy.Module.load_from_file(device, SPLATGS_RENDER_SHADER)


def load_adc_module(device: spy.Device):
    return spy.Module.load_from_file(device, SPLATGS_ADC_SHADER)
