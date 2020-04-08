from datetime import datetime
import logging
import progressbar
from sca.util import progress_bar_util


class LogHandler:
    PATH = "out/"
    CONST_SEPARATOR = "____________________________________________\n"

    def __init__(self, source: str, debug_mode_enabled: bool):
        """
        Constructor for the log_handler, adds a file log_handler

        :param source: source path to log to
        :param debug_mode_enabled: Whether debug mode is enabled, whether to log
        """

        if debug_mode_enabled:
            # How to format our data
            formatter = logging.Formatter('%(message)s')

            # Create File logger
            self.file_logger = logging.getLogger(source)
            self.file_logger.setLevel("DEBUG")

            # Create file handler
            self.file_name = self.PATH + 'log-' + str(source) + '-' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.log'
            self.file_handler = logging.FileHandler(self.file_name)
            self.file_handler.setLevel("DEBUG")
            self.file_handler.setFormatter(formatter)
            self.file_handler.delay = True

            # Add the handler to the logger
            self.file_logger.addHandler(self.file_handler)

            # Print to the console that debug mode has been enabled
            print("Debug mode is enabled. Detailed information will be outputted to: " + self.file_name)

        self.debug_mode_enabled = debug_mode_enabled

    def log_to_debug_file(self, data: str):
        """
        Logs data to the log file
        :param data: data to log
        """

        if self.debug_mode_enabled:
            self.file_logger.debug(data)

    def log_to_debug_progressbar(self, bar: progressbar.progressbar, data: str):
        """
        Logs data to the progressbar

        :param data: data to log
        :param bar: the progressbar to log to
        """

        if self.debug_mode_enabled:
            progress_bar_util.set_debug_message(bar, data)

    def stop_logging(self):
        """
        Stops the log handler from working, closes the file. Can be called after an attack for example, when the
        file should be released and the debugging is done as well.
        """
        self.file_handler.close()
        self.file_logger.handlers.clear()
        self.debug_mode_enabled = False
