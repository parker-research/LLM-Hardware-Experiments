import os
import sys
import platform
from datetime import datetime, timezone
from typing import Any
import orjson
from pathlib import Path

from loguru import logger


def _get_system_environment_info() -> dict[str, dict[str, Any]]:

    # Create a dictionary to store environment info
    env_info = {
        "System Info": {
            "Platform": platform.platform(),
            "System": platform.system(),
            "Machine": platform.machine(),
            "Processor": platform.processor(),
            "Python Version": platform.python_version(),
            "Python Implementation": platform.python_implementation(),
        },
        "Environment Info": {
            "Working Directory": os.getcwd(),
            "Python Executable": sys.executable,
            "Python Path": sys.path,
            "Python Prefix": sys.prefix,
            "Python Exec Prefix": sys.exec_prefix,
            "Python Version Info": list(sys.version_info),
            "Python Build Info": sys.version,
        },
        # TODO: consider adding back certain environment variables; too dangerous to log all though
        # "Environment Variables": {key: value for key, value in os.environ.items()},
        "Process Info": {
            "Process ID": os.getpid(),
            "Parent Process ID": os.getppid(),
            "Process Group ID": os.getpgrp(),
            "Process User ID": os.getuid(),
            "Process Effective User ID": os.geteuid(),
            "Process Group ID (again)": os.getgid(),
            "Process Effective Group ID": os.getegid(),
        },
        "Other Info": {
            "Current Local Time": datetime.now(),
            "Current UTC Time": datetime.now(timezone.utc),
            "Timezone": str(datetime.now().astimezone().tzinfo),
            "Timezone Name": datetime.now().astimezone().tzname(),
        },
    }
    return env_info


def _get_gpu_env_info():
    """Logs GPU information."""
    gpu_info = {}

    # Attempt to import and log PyTorch GPU info
    try:
        import torch

        try:
            cuda_current_device = torch.cuda.current_device()
        except RuntimeError:
            cuda_current_device = "RuntimeError: No CUDA devices found."

        pytorch_info = {
            "PyTorch Version": torch.__version__,
            "PyTorch CUDA Is Available": torch.cuda.is_available(),
            "PyTorch CUDA Device Count": torch.cuda.device_count(),
            "PyTorch CUDA Current Device": cuda_current_device,
            "PyTorch CUDA Devices": [
                {
                    "Device Number": cuda_dev_num,
                    "Name": torch.cuda.get_device_name(cuda_dev_num),
                    "Capability": str(torch.cuda.get_device_capability(cuda_dev_num)),
                    "Device Properties": str(torch.cuda.get_device_properties(cuda_dev_num)),
                }
                for cuda_dev_num in range(torch.cuda.device_count())
            ],
        }
        gpu_info["PyTorch GPU Info"] = pytorch_info
    except ImportError:
        logger.warning("PyTorch not installed; skipping GPU info.")
        gpu_info["PyTorch GPU Info"] = "PyTorch not installed"

    # Attempt to import and log TensorFlow GPU info
    try:
        import tensorflow as tf

        tensorflow_info = {
            "TensorFlow Version": tf.__version__,
            "TensorFlow Built with CUDA": tf.test.is_built_with_cuda(),
            # "TensorFlow CUDA GPU available": tf.test.is_gpu_available(cuda_only=True), # dep
            "TensorFlow CUDA GPU Devices": tf.config.list_physical_devices("GPU"),
        }
        gpu_info["TensorFlow GPU Info"] = tensorflow_info
    except ImportError:
        logger.warning("TensorFlow not installed; skipping GPU info.")
        gpu_info["TensorFlow GPU Info"] = "TensorFlow not installed"

    return gpu_info


def get_all_env_info() -> dict[str, dict[str, Any]]:
    env_info = _get_system_environment_info()
    gpu_info = _get_gpu_env_info()

    env_info.update(gpu_info)
    return env_info


def log_env_info(env_info: dict[str, dict[str, Any]]):
    """Logs environment information."""

    logger.info("Logging environment info...")
    for section, values in env_info.items():
        logger.info(f"{section}:")
        if isinstance(values, dict):
            for key, value in values.items():
                logger.info(f"{key}: {value}")
        else:
            logger.info(values)

    logger.info("Finished logging environment info.")


def write_env_info_to_json_file(env_info: dict[str, dict[str, Any]], json_file_path: Path) -> None:
    with open(json_file_path, "wb") as f:
        f.write(orjson.dumps(env_info, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    env_info = get_all_env_info()
    log_env_info(env_info)

    from llm_experiments.util.path_helpers import make_data_dir

    working_dir = make_data_dir("testing", append_date=False)  # in <git_root>/working/testing
    working_dir.mkdir(parents=True, exist_ok=True)
    json_file_path = working_dir / "env_info.json"
    write_env_info_to_json_file(env_info, json_file_path)

    logger.info(f"Environment info written to {json_file_path}.")
