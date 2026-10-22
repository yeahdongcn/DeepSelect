import enum
import functools
import glob
import subprocess
import os


class Platform(enum.Enum):
    CUDA = "CUDA"
    ASCEND = "ASCEND"
    MUSA = "MUSA"
    CPU_ONLY = "CPU_ONLY"


@functools.lru_cache(maxsize=1)
def get_current_platform() -> Platform:
    """
    Get the current platform via device nodes and `lspci`
    """
    if os.path.exists("/dev/davinci_manager"):
        return Platform.ASCEND
    # MUSA keeps `torch.cuda.is_available()` False on purpose, so the device node is the
    # probe -- the same one `setup.py` and `deep_select/interface.py` use. On S5000 the node
    # is `/dev/mtgpu.0` (with a dot), so this has to be a glob.
    if glob.glob("/dev/mtgpu*") or glob.glob("/dev/musa*"):
        return Platform.MUSA
    # `lspci` is absent from MUSA containers, and this module is imported by every test on
    # every platform, so a missing binary must not abort the suite.
    try:
        output = subprocess.check_output(["lspci"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return Platform.CPU_ONLY
    if "3D controller: NVIDIA Corporation Device" in output:
        return Platform.CUDA
    return Platform.CPU_ONLY


def assert_current_platform(target_platform: Platform | list[Platform]):
    """
    Assert that the current platform matches the expected platform(s).

    Args:
        target_platform: A single Platform or a list of Platforms to check against.

    Raises:
        RuntimeError: If the current platform is not in the target list.
    """
    if isinstance(target_platform, Platform):
        target_platform = [target_platform]
    current_platform = get_current_platform()
    if current_platform not in target_platform:
        raise RuntimeError(
            f"Current platform is {current_platform.value}, but expected {[p.value for p in target_platform]}"
        )


def requires_platform(target_platform: Platform | list[Platform]):
    """
    Decorator that ensures the current platform matches before the function is called.

    Args:
        target_platform: A single Platform or a list of Platforms that are allowed.

    Raises:
        RuntimeError: If the current platform does not match.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            assert_current_platform(target_platform)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def is_on_cuda_platform() -> bool:
    return get_current_platform() == Platform.CUDA


def is_on_ascend_platform() -> bool:
    return get_current_platform() == Platform.ASCEND


def is_on_musa_platform() -> bool:
    return get_current_platform() == Platform.MUSA


def is_on_cpu_only_platform() -> bool:
    return get_current_platform() == Platform.CPU_ONLY
