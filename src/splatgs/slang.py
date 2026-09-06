import pathlib

import torch
import slangpy as spy


SPLATGS_SHADER_PATH = str(pathlib.Path(__file__).parent / "shaders")
"""Path to `splatgs` shader directory"""
SPLATGS_RENDER_SHADER = "render.slang"
"""Path to `splatgs` render shader"""
SPLATGS_ADC_SHADER = "adc.slang"
"""Path to `splatgs` adc shader"""


# This function is essentialy a reworked version of slangpy.create_torch_device
def create_slang_device(
        type: spy.DeviceType=spy.DeviceType.cuda,
        include_paths=[],
        torch_device=None,
        fp_mode: spy.SlangFloatingPointMode=spy.SlangFloatingPointMode.default,
        optimization_level: spy.SlangOptimizationLevel=spy.SlangOptimizationLevel.default,
        debug_info: spy.SlangDebugInfoLevel = spy.SlangDebugInfoLevel.standard,
        enable_debug_layers: bool=False,
        enable_print: bool=False):
    """Create SlangPy device with PyTorch integration

    - type: SlangPy device type (CUDA, Vulkan...)
    - include_paths: additional search paths for Slang code
    - torch_device: Torch device to share context with
    - fp_mode: floating-point mode for Slang code compilation
    - optimization_level: optimization level for Slang code compilation
    - debug_info: include debug information during Slang code compilation
    - enable_debug_layers: enable debug/validation layers
    - enable_print: enable print in Slang code
    """

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
            "debug_info" : debug_info,
        },
        enable_debug_layers=enable_debug_layers,
        enable_print=enable_print,
        enable_cuda_interop=(type != spy.DeviceType.cuda),
        existing_device_handles=handles,
    )


def load_render_module(device: spy.Device):
    """Load `splatgs` render module using `device`"""
    return spy.Module.load_from_file(device, SPLATGS_RENDER_SHADER)


def load_adc_module(device: spy.Device):
    """Load `splatgs` adc module using `device`"""
    return spy.Module.load_from_file(device, SPLATGS_ADC_SHADER)
