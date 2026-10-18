"""
Hardware-adaptive runtime profile.

The same code base runs on a 15 W laptop CPU and on a datacenter GPU; only
the compute budget changes. Detection is stdlib-only and every value can be
overridden with an environment variable.

Profiles
    gpu      NVIDIA GPU found: full-resolution images, all three tiers.
    cpu      No GPU: threads = physical cores - 1 (keeps the machine usable),
             768px images, and the 7B tier is OPT-IN (ALLOW_LARGE_TIER=1)
             because one 7B vision call on a laptop CPU takes many minutes.

Overrides
    OLLAMA_NUM_THREAD, MAX_IMAGE_SIDE, REQUEST_TIMEOUT, ALLOW_LARGE_TIER=0/1,
    USE_TEXT_TIER=0/1, SOVEREIGN_PROFILE=cpu|gpu
"""

import os
import shutil


def _has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def _physical_cores() -> int:
    # Assumes 2-way SMT (true for Intel Core i3/i5/i7 and Ryzen); override if wrong.
    return max(1, (os.cpu_count() or 2) // 2)


def _env_flag(name, default):
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes")


def detect_profile() -> dict:
    name = os.environ.get("SOVEREIGN_PROFILE") or ("gpu" if _has_nvidia_gpu() else "cpu")
    if name == "gpu":
        profile = {"name": "gpu", "num_thread": None, "max_image_side": 1280,
                   "timeout": 180, "allow_large": True, "use_text_tier": True}
    else:
        profile = {"name": "cpu", "num_thread": max(1, _physical_cores() - 1),
                   "max_image_side": 768, "timeout": 900, "allow_large": False,
                   "use_text_tier": True}

    if os.environ.get("OLLAMA_NUM_THREAD"):
        profile["num_thread"] = int(os.environ["OLLAMA_NUM_THREAD"])
    if os.environ.get("MAX_IMAGE_SIDE"):
        profile["max_image_side"] = int(os.environ["MAX_IMAGE_SIDE"])
    if os.environ.get("REQUEST_TIMEOUT"):
        profile["timeout"] = int(os.environ["REQUEST_TIMEOUT"])
    profile["allow_large"] = _env_flag("ALLOW_LARGE_TIER", profile["allow_large"])
    profile["use_text_tier"] = _env_flag("USE_TEXT_TIER", profile["use_text_tier"])
    return profile


PROFILE = detect_profile()


if __name__ == "__main__":
    print(PROFILE)
