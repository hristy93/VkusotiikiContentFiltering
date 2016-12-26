# -*- coding: utf-8 -*-

""" Enables the unicode console for windows """
def enable_win_unicode_console():
    try:
        # Fix UTF8 output issues on Windows console.
        # Does nothing if package is not installed
        from win_unicode_console import enable
        enable()
    except ImportError:
        pass


if __name__ == "__main__":
    if sys.platform == "win32":
        enable_win_unicode_console()