import unittest
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import logging
import structlog
from src import logging_config


class LoggingTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(TemporaryDirectory()))

    def test_logging_reconfigures_destinations_without_resetting_host(self):
        tmp_path = self.tmp_path
        project = logging.getLogger("calmsense")
        saved = (project.handlers[:], project.level, project.propagate)
        project.handlers = []
        host = logging.getLogger()
        host_handler = logging.NullHandler()
        host.addHandler(host_handler)
        structlog_before = structlog.get_config().copy()
        self.stack.enter_context(
            patch.object(logging_config, "_configured_settings", None)
        )
        first, second = (tmp_path / "first.log", tmp_path / "second.log")
        try:
            logging_config.setup_logging(log_file=first, console=False)
            logger = logging_config.get_logger("src.test")
            logger.info("first", value=1)
            old_handler = project.handlers[0]
            logging_config.setup_logging(log_file=second, console=False)
            logger.info("second", value=2)
            self.assertIs(old_handler.stream, None)
            self.assertEqual(json.loads(first.read_text())["event"], "first")
            self.assertEqual(json.loads(second.read_text())["event"], "second")
            self.assertIn(host_handler, host.handlers)
            self.assertEqual(structlog.get_config(), structlog_before)
        finally:
            for handler in project.handlers:
                handler.close()
            project.handlers, project.level, project.propagate = saved
            host.removeHandler(host_handler)

    def test_invalid_logging_level_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown logging level"):
            logging_config.setup_logging(level="invalid", log_file=None)

    def test_failed_reconfiguration_keeps_the_working_logger(self):
        tmp_path = self.tmp_path
        project = logging.getLogger("calmsense")
        saved = (project.handlers[:], project.level, project.propagate)
        project.handlers = []
        self.stack.enter_context(
            patch.object(logging_config, "_configured_settings", None)
        )
        destination = tmp_path / "working.log"
        blocker = tmp_path / "file-not-directory"
        blocker.write_text("keep")
        try:
            logging_config.setup_logging(log_file=destination, console=False)
            handler = project.handlers[0]
            settings = logging_config._configured_settings
            logger = logging_config.get_logger("src.test")
            logger.info("before")
            with self.assertRaises(OSError):
                logging_config.setup_logging(
                    level="DEBUG", log_file=blocker / "new.log"
                )
            self.assertEqual(project.handlers, [handler])
            self.assertEqual(project.level, logging.INFO)
            self.assertEqual(logging_config._configured_settings, settings)
            self.assertIsNot(handler.stream, None)
            logger.info("after failure")
            logging_config.setup_logging(log_file=destination, console=False)
            logger.info("after retry")
            self.assertEqual(
                [
                    json.loads(line)["event"]
                    for line in destination.read_text().splitlines()
                ],
                ["before", "after failure", "after retry"],
            )
        finally:
            for handler in project.handlers:
                handler.close()
            project.handlers, project.level, project.propagate = saved
