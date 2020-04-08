import unittest
import progressbar
from sca.util import progress_bar_util


class TestProgressbarUtil(unittest.TestCase):

    def test_set_debug_message(self):
        """
        Tests if a debug message is printed next to the progressbar
        """

        test_string = "ProgressBarProgressBar"

        bar = progressbar.ProgressBar(widgets=progress_bar_util.CONST_DEFAULT_DEBUG_WIDGETS)
        progress_bar_util.set_debug_message(bar, "ProgressBarProgressBar")

        desired_value = {progress_bar_util.CONST_DEBUG_STRING: test_string}
        self.assertEqual(bar.dynamic_messages, desired_value)

    def test_get_widgets_debug(self):
        """
        Tests if the debug widgets are returned correctly
        """

        real_widgets = progress_bar_util.CONST_DEFAULT_DEBUG_WIDGETS
        result = progress_bar_util.get_widgets(True)

        self.assertEqual(real_widgets, result)

    def test_get_widgets(self):
        """
        Tests if the normal widgets are returned correctly
        """

        real_widgets = progress_bar_util.CONST_DEFAULT_WIDGETS
        result = progress_bar_util.get_widgets(False)

        self.assertEqual(real_widgets, result)
