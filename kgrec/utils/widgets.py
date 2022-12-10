from collections import Sequence

from progressbar import Counter, Bar, Timer, ETA

_default_widgets = [
    ' ',
    Counter(format='%(value)02d/%(max_value)d'),
    Bar(),
    ' ',
    ' [', Timer(), '] ',
    ' (', ETA(), ') '
]


def widgets_with_label(label: str) -> Sequence[object]:
    """
    gets widgets for a progress bar with the specified label as a suffix.

    :param label: that shall be written as suffix to the progress bar.
    :return: a list of widgets for a progress bar.
    """
    widgets = [label]
    for w in _default_widgets:
        widgets.append(w)
    return widgets
