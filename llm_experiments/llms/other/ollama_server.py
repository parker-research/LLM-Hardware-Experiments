import subprocess
import os
import signal
import re
from datetime import datetime
import time
from pathlib import Path
import uuid

from loguru import logger

from llm_experiments.util.path_helpers import (
    make_data_dir,
    get_file_date_str,
    get_folder_total_size,
)
from llm_experiments.util.download_helpers import download_large_file
from llm_experiments.util.execute_cli import run_command


_ollama_server_singleton = None


def start_ollama_server(log_file_path: Path | None = None):
    global _ollama_server_singleton
    if _ollama_server_singleton is None:
        _ollama_server_singleton = _OllamaServer()
    if log_file_path is not None:
        _ollama_server_singleton.set_log_file_path(log_file_path)
    _ollama_server_singleton.start()


def set_ollama_server_log_file_path(log_file_path: Path):
    global _ollama_server_singleton
    if _ollama_server_singleton is None:
        logger.warning("Ollama server not started yet.")
    else:
        _ollama_server_singleton.set_log_file_path(log_file_path)


def stop_ollama_server():
    global _ollama_server_singleton
    if _ollama_server_singleton is None:
        logger.warning("Ollama server not started yet.")
    else:
        _ollama_server_singleton.stop()


def get_ollama_folder_size_bytes() -> int | None:
    global _ollama_server_singleton
    if _ollama_server_singleton is None:
        logger.warning("Ollama server not started yet.")
        return None
    else:
        return get_folder_total_size(_ollama_server_singleton._ollama_model_data_store_path)


class _OllamaServer:
    def __init__(self, log_file_path: Path | None = None):
        self.process = None
        self.log_file_path = log_file_path

        self._ollama_executable_path = (
            make_data_dir(f"tools/ollama/ollama_{datetime.now().date()}", append_date=False)
            / "ollama-linux-amd64"
        )

        # Set the path to the Ollama model data store to avoid re-downloading models as much.
        # Gets passed to "OLLAMA_MODELS" environment variable.
        self._ollama_model_data_store_path = make_data_dir(
            "tools/ollama_model_data", append_date=False
        )

        # NOTE: other config options from "./ollama serve --help" include:
        # OLLAMA_HOST         The host:port to bind to (default "127.0.0.1:11434")
        # OLLAMA_ORIGINS      A comma separated list of allowed origins.
        # OLLAMA_MODELS       The path to the models directory (default is "~/.ollama/models")
        # OLLAMA_KEEP_ALIVE   The duration that models stay loaded in memory (default is "5m")

    def set_log_file_path(self, log_file_path: Path) -> None:
        self.log_file_path = log_file_path

    def start(self):
        """Starts the ollama serve command in the background."""
        if self.process:
            logger.warning("Ollama server is already running.")
            return

        self._download_ollama()

        # Using subprocess.Popen to execute the command in the background
        if self.log_file_path is None:
            self.log_file_path = (
                make_data_dir("orphan_experiment/ollama_server_logs", append_date=False)
                / f"ollama_serve_{get_file_date_str()}.log"
            )
        self._log_file_pointer = self.log_file_path.open("w")
        self.process = subprocess.Popen(
            [str(self._ollama_executable_path), "serve"],
            stdout=self._log_file_pointer,
            stderr=subprocess.STDOUT,  # redirect stderr to stdout file
            text=True,
            preexec_fn=os.setsid,
            env=dict(os.environ) | {"OLLAMA_MODELS": str(self._ollama_model_data_store_path)},
        )  # Set session ID
        logger.info(f"Ollama server started with PID: {self.process.pid}")

        # check on the process's log file, and make sure it's running okay
        has_warned_process_terminated = False
        has_warned_address_in_use = False
        for _ in range(10):  # 10 secs
            if (self.process.poll() is not None) and (not has_warned_process_terminated):
                logger.warning(
                    f"Ollama server process terminated with return code: {self.process.returncode}"
                )
                has_warned_process_terminated = True

            # process is still running, check log
            with self.log_file_path.open("r") as log_file:
                log_contents = log_file.read()
                if ("address already in use" in log_contents) and (not has_warned_address_in_use):
                    logger.warning("Ollama got 'address already in use' error.")
                    has_warned_address_in_use = True

            time.sleep(1)
            if has_warned_process_terminated and has_warned_address_in_use:
                break

        logger.info(f"Ollama server log contents:\n{log_contents}")

    def stop(self):
        """Stops the ollama serve command if it is running."""
        if self.process:
            if (ret_code := self.process.poll()) is not None:
                logger.warning(
                    "Trying to stop Ollama server, but "
                    f"Ollama server process already terminated with return code: {ret_code}"
                )
            else:
                # Sending SIGTERM to the process group to ensure clean shutdown
                logger.info(f"Stopping Ollama server (PID={self.process.pid})...")
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait()  # Wait for the process to terminate
                logger.info("Ollama server stopped.")
        else:
            logger.info("Ollama server is not running (process not set).")

        try:
            self._log_file_pointer.close()
        except Exception as e:
            logger.info(f"Tried to close 'ollama serve' log file, but got error: {e}.")
        else:
            logger.info("Closed 'ollama serve' log file.")

    def _download_ollama(self) -> None:
        temp_download_file_path = self._ollama_executable_path.with_suffix(
            f".tmp_download_{uuid.uuid4()}"
        )
        if not self._ollama_executable_path.is_file():
            # TODO: refactor temp file deal into download_large_file
            logger.info("Downloading Ollama...")
            download_large_file(
                "https://ollama.com/download/ollama-linux-amd64",
                temp_download_file_path,
            )
            temp_download_file_path.rename(self._ollama_executable_path)
            logger.info(f"Downloaded Ollama to {self._ollama_executable_path}")

        else:
            logger.info(f"Ollama already downloaded to {self._ollama_executable_path}")

        self._ollama_executable_path.chmod(0o755)
        logger.info(f"Ollama version: {self._get_ollama_version()}")

    def get_ollama_version(self) -> str | None:
        # ensure Ollama is downloaded
        self._download_ollama()

        # get the version
        return self._get_ollama_version()

    def _get_ollama_version(self) -> str | None:
        """Gets Ollama version by running the executable. Assumes executable is downloaded."""
        assert (
            self._ollama_executable_path.is_file()
        ), f"Ollama executable not found: {self._ollama_executable_path}"
        command_result = run_command([str(self._ollama_executable_path), "--version"])

        assert command_result.return_code == 0, f"Error getting Ollama version: {command_result}"

        version_num_match: str = re.search(
            r"version is (?P<version_num>(\d+\.?)+)", command_result.stdout, re.IGNORECASE
        )
        if version_num_match is None:
            logger.warning(
                f"Could not find Ollama version number from stdout: {command_result.stdout}"
            )
            return None
        else:
            return version_num_match.group("version_num")


if __name__ == "__main__":
    start_ollama_server()
    input("Press Enter to stop the server...")
    stop_ollama_server()
