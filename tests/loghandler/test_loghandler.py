import unittest
from sca.loghandler import loghandler
import os
import progressbar
from sca.util import progress_bar_util


class TestLogHandler(unittest.TestCase):

    def test_create_handler_debug_mode_enabled(self):
        """"This tests whether creating a debug mode log handler creates a file handler"""

        log_handler = loghandler.LogHandler("test_file", True)
        file_name = log_handler.file_name
        log_handler.stop_logging()

        # Delete the file after creation
        os.remove(file_name)

        self.assertTrue(log_handler.file_handler is not None)

    def test_create_handler_debug_mode_disabled(self):
        """"This tests whether creating a non debug mode log handler works and does not create a file"""

        log_handler = loghandler.LogHandler("test_file2", False)
        self.assertFalse(hasattr(log_handler, "file_name"))

    def test_create_file_debug_mode_enabled(self):
        """"This tests whether creating a debug mode log handler creates a file"""

        log_handler = loghandler.LogHandler("test_file1", True)
        file_name = log_handler.file_name
        log_handler.stop_logging()

        self.assertTrue(os.path.isfile(file_name))

        # Delete the file after creation
        os.remove(file_name)

    def test_log_to_debug_file(self):
        """"This tests whether loghandler to a file works"""

        # Create a log handler and write something, then close it.
        log_handler = loghandler.LogHandler("test_file3", True)
        file_name = log_handler.file_name
        test_string = "Test123"
        log_handler.log_to_debug_file(test_string)
        log_handler.stop_logging()

        test_file = open(file_name)

        # When closing the file a newline is printed
        self.assertEqual(test_file.read(), test_string + "\n")

        test_file.close()

        # Delete the file after creation
        os.remove(file_name)

    def test_log_to_debug_progressbar(self):
        """"This tests whether loghandler to the progressbar works"""

        log_handler = loghandler.LogHandler("test_file4", True)
        test_string = "Test123"

        # Create a progressbar and debug a certain test string
        log_bar = progressbar.ProgressBar(widgets=progress_bar_util.get_widgets(True))
        log_handler.log_to_debug_progressbar(log_bar, test_string)

        log_handler.stop_logging()

        # Test if the debug message while loghandler is equal to the desired string
        desired_value = {progress_bar_util.CONST_DEBUG_STRING: test_string}
        self.assertEqual(log_bar.dynamic_messages, desired_value)

        os.remove(log_handler.file_name)

    def test_log_file_forbidden(self):
        """Tests whether trying to log to a non existent file does not work"""
        log_handler = loghandler.LogHandler("test_file7", False)
        log_handler.log_to_debug_file("test123")
        self.assertFalse(hasattr(log_handler, "file_name"))

    def test_log_progressbar_forbidden(self):
        """Tests whether trying to log to a progressbar while not in debug mode does not work"""
        log_handler = loghandler.LogHandler("test_file8", False)

        log_bar = progressbar.ProgressBar(widgets=progress_bar_util.get_widgets(True))
        log_handler.log_to_debug_progressbar(log_bar, "test123")

        # Test if the debug message while logging is equal to the desired string
        desired_value = {progress_bar_util.CONST_DEBUG_STRING: None}
        self.assertEqual(log_bar.dynamic_messages, desired_value)

    def test_stop_logging_to_file(self):
        """Tests whether the stop_logging function stops the loghandler to a file"""

        # Create a log handler and write something, then close it.
        log_handler = loghandler.LogHandler("test_file5", True)
        file_name = log_handler.file_name
        test_string = "Test123"
        log_handler.log_to_debug_file(test_string)
        log_handler.stop_logging()

        # Write again a message which shouldnt be logged
        test_string2 = "Test456"
        log_handler.log_to_debug_file(test_string2)

        test_file = open(file_name)

        # Test if only test_string has been logged
        self.assertEqual(test_file.read(), test_string + "\n")

        test_file.close()

        # Delete the file after creation
        os.remove(file_name)

    def test_stop_logging_to_progressbar(self):
        """Tests whether the stop_logging function stops the loghandler to the progressbar"""

        log_handler = loghandler.LogHandler("test_file6", True)
        test_string = "Test123"

        # Create a progressbar and debug a certain test string
        log_bar = progressbar.ProgressBar(widgets=progress_bar_util.get_widgets(True))
        log_handler.log_to_debug_progressbar(log_bar, test_string)
        log_handler.stop_logging()

        # Write again a message which shouldnt be logged
        test_string2 = "Test456"
        log_handler.log_to_debug_progressbar(log_bar, test_string2)

        # Test if the debug message while loghandler is equal to the desired string
        desired_value = {progress_bar_util.CONST_DEBUG_STRING: test_string}
        self.assertEqual(log_bar.dynamic_messages, desired_value)

        os.remove(log_handler.file_name)
