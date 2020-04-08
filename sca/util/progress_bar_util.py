import progressbar
from typing import List

CONST_DEBUG_STRING = "debug_mode"
CONST_DEBUG_WIDTH = 1
CONST_DEBUG_PRECISION = 150

CONST_DEFAULT_WIDGETS = ['[', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ')', ' (',
                         progressbar.Percentage(), ')']

CONST_DEFAULT_DEBUG_WIDGETS = ['(', progressbar.ETA(), ')', ' (', progressbar.Percentage(), ') ',
                               progressbar.DynamicMessage(name=CONST_DEBUG_STRING, width=CONST_DEBUG_WIDTH,
                                                          precision=CONST_DEBUG_PRECISION)]


def set_debug_message(bar: progressbar.bar, message: str):
    """ Sets a debug message to be printed next to the progressbar.

    :param bar: The progressbar to use.
    :param message: The debug message to print.
    """

    bar.dynamic_messages[CONST_DEBUG_STRING] = message


def get_widgets(debug_mode_enabled: bool) -> List:
    """ Gets the correct widgets for the progressbar to be rendered.

    :param debug_mode_enabled: Whether debug_mode is on or off.
    :return: a list of widgets to be rendered on the progressbar.
    """

    if debug_mode_enabled:
        return CONST_DEFAULT_DEBUG_WIDGETS
    else:
        return CONST_DEFAULT_WIDGETS
