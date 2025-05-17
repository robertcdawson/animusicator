import os
from loguru import logger


def test_log_file_creation(tmp_path):
    log_file = tmp_path / "test.log"

    # Configure logger to write only to the temporary file
    logger.remove()
    logger.add(log_file, format="{level}: {message}", level="DEBUG")

    logger.debug("debug message")
    logger.info("info message")
    logger.success("success message")
    logger.warning("warning message")
    logger.error("error message")

    try:
        1 / 0
    except Exception:
        logger.exception("An exception occurred")

    # Remove handlers to avoid side effects on other tests
    logger.remove()

    assert log_file.exists()
    contents = log_file.read_text()
    for message in [
        "debug message",
        "info message",
        "success message",
        "warning message",
        "error message",
        "An exception occurred",
    ]:
        assert message in contents
