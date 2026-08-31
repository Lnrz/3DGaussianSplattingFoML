import sys
from pathlib import Path

import slangpy as spy


ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"


# make splatgs importable
sys.path.append(str(SRC_DIR))


def resolve_data_path(path: str | Path):
    if not isinstance(path, Path):
        input_path = Path(path)

    if input_path.exists():
        return input_path

    path_in_data_dir = DATA_DIR / input_path
    if path_in_data_dir.exists():
        return path_in_data_dir

    matches = list(DATA_DIR.rglob(path))
    if not matches:
        raise FileNotFoundError(f"Found no match in 'data' directory for '{path}'")
    elif len(matches) > 2:
        err_msg = f"Found too many matches in 'data' directory for '{path}':\n"
        for match in matches:
            err_msg += 4*" " + str(match.relative_to(DATA_DIR)) + "\n"
        raise ValueError(err_msg)

    return matches[0]


_slang_fp_mode_str_to_enum_dict = {
    "fast" : spy.SlangFloatingPointMode.fast,
    "default" : spy.SlangFloatingPointMode.default,
    "precise" : spy.SlangFloatingPointMode.precise
}
_slang_fp_mode_enum_to_str_dict = {enum : string for string, enum in _slang_fp_mode_str_to_enum_dict.items()}

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


_slang_optim_str_to_enum_dict = {
    "none" : spy.SlangOptimizationLevel.none,
    "default" : spy.SlangOptimizationLevel.default,
    "high" : spy.SlangOptimizationLevel.high,
    "maximal" : spy.SlangOptimizationLevel.maximal
}
_slang_optim_enum_to_str_dict = {enum : string for string, enum in _slang_optim_str_to_enum_dict.items()}

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