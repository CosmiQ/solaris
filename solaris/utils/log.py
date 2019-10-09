import logging


def _get_logging_level(level_int):
    """Convert a logging level integer into a log level."""
    if isinstance(level_int, bool):
        level_int = int(level_int)
    if level_int < 0:
        return logging.CRITICAL + 1  # silence all possible outputs
    elif level_int == 0:
        return logging.WARNING
    elif level_int == 1:
        return logging.INFO
    elif level_int == 2:
        return logging.DEBUG
    elif level_int in [10, 20, 30, 40, 50]:  # if user provides the logger int
        return level_int
    elif isinstance(level_int, int):  # if it's an int but not one of the above
        return level_int
    else:
        raise ValueError(f"logging level set to {level_int}, "
                         "but it must be an integer <= 2.")
