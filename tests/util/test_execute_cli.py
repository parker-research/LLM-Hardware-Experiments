from datetime import timedelta

from llm_experiments.util.execute_cli import run_command


def test_run_command__basic():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.stdout.strip() == "hello"
    assert result.stderr == ""
    assert result.timed_out is False
    assert 0 < result.execution_duration.total_seconds() < 0.5


def test_run_command__basic_sleep():
    result = run_command(["sleep", "1s"])
    assert result.return_code == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert result.timed_out is False
    assert 1 < result.execution_duration.total_seconds() < 1.1


def test_run_command__basic_sleep_not_timed_out():
    result = run_command(["sleep", "1s"], max_execution_time=timedelta(seconds=3))
    assert result.return_code == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert result.timed_out is False
    assert 1 < result.execution_duration.total_seconds() < 1.1


def test_run_command__basic_sleep_timed_out():
    result = run_command(["sleep", "1s"], max_execution_time=timedelta(seconds=0.5))
    assert result.return_code is None  # None when timed out
    assert result.stdout == ""
    assert result.stderr == ""
    assert result.timed_out is True
    assert 0.5 < result.execution_duration.total_seconds() < 0.7  # approx max_execution_time


# TODO: add test to check that the output is still captured when the command times out
