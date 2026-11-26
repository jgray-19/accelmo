"""``configure_logging`` splits the terminal and file levels."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from accelmo.logs import configure_logging

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_console_gets_warnings_and_the_file_gets_info(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    try:
        configure_logging(tmp_path / "logs" / "run.log")
        logging.getLogger("accelmo.test_logs").info("setup done")
        logging.getLogger("accelmo.test_logs").warning("odd value")
        logging.getLogger("accelmo.test_logs").debug("detail")
        for handler in root.handlers:
            handler.flush()
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved:
            root.addHandler(handler)
        root.setLevel(level)
    err = capsys.readouterr().err
    assert "odd value" in err
    assert "setup done" not in err
    text = (tmp_path / "logs" / "run.log").read_text()
    assert "setup done" in text
    assert "odd value" in text
    assert "detail" not in text
