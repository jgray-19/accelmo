"""Workers log to their own file, and only their warnings reach the parent."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from typing import TYPE_CHECKING

from accelmo.fitting.worker_logging import attach_in_worker, start_relay

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

WORKER_LOGGER = "accelmo.test_worker_logging"
RELAY_WAIT_SECONDS = 10.0


def _worker(logfile: Path) -> None:
    attach_in_worker(logfile, logging.DEBUG)
    log = logging.getLogger(WORKER_LOGGER)
    log.debug("detail %d", 7)
    log.info("an info line")
    log.warning("lost %s", "particles")


def test_worker_file_has_everything_and_only_warnings_reach_the_parent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    logfile = tmp_path / "workers" / "python_worker_0.log"
    start_relay()
    with caplog.at_level(logging.DEBUG):
        process = mp.get_context("fork").Process(target=_worker, args=(logfile,), name="W-0")
        process.start()
        process.join(timeout=30)
        assert process.exitcode == 0
        deadline = time.monotonic() + RELAY_WAIT_SECONDS
        while time.monotonic() < deadline and not any(
            r.name == WORKER_LOGGER for r in caplog.records
        ):
            time.sleep(0.05)
    text = logfile.read_text()
    assert "detail 7" in text
    assert "an info line" in text
    assert "W-0" in text
    relayed = [r for r in caplog.records if r.name == WORKER_LOGGER]
    assert [r.getMessage() for r in relayed] == ["W-0: lost particles"]
    assert relayed[0].levelno == logging.WARNING
