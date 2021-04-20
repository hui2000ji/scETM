import logging
import functools
import os

_logger = logging.getLogger(__name__)


def log_arguments(f):
    """Logs the arguments of the function on calling.

    Args:
        f: the decorated function whose arguments will be logged.

    Returns:
        log_arguments_wrapper: a wrapped function that will log the arguments
            on calling.
    """

    if hasattr(f, '__qualname__'):
        name = f.__qualname__
    else:
        name = f.__name__
    @functools.wraps(f)
    def log_arguments_wrapper(*args, **kwargs):
        if name.endswith('__init__'):
            assert len(args) >= 1
            args_str = ', '.join(map(str, args[1:]))
        else:
            args_str = ', '.join(map(str, args))
        kwargs_str = ', '.join([f'{k} = {v}' for k, v in kwargs.items()])
        args_kwargs_str = ', '.join([s for s in [args_str, kwargs_str] if s])
        _logger.info(f'{name}({args_kwargs_str})')
        result = f(*args, **kwargs)
        return result
    return log_arguments_wrapper


def initialize_logger(ckpt_dir=None, level=logging.INFO, logger=None) -> None:
    """Initializes the scETM logger, or the provided logger.

    Each time the function is called, close all file handlers of the scETM
    logger and create one logging to "ckpt_dir/log.txt" if ckpt_dir is not
    None.
    If the scETM logger has no handlers, create a formatted stream handler.

    Args:
        ckpt_dir: directory to store the log file. If None, no file handler
            will be added to the scETM logger.
        level: controls the level of logging.
        logger: the provided logger to be configured.
    """
    if logger is None:
        logger = logging.getLogger('scETM')
    logger.setLevel(level)
    if logger.hasHandlers():
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.warning(f'Reinitializing... The file handler {handler} will be closed.')
                logger.removeHandler(handler)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s')
    if not logger.hasHandlers():
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)
    if ckpt_dir is not None:
        file_handler = logging.FileHandler(os.path.join(ckpt_dir, 'log.txt'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
