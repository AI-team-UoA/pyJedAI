import logging
import sys
from logging import WARN  # noqa

_interactive = False
try:
    # This is only defined in interactive shells
    if sys.ps1:
        _interactive = True
except AttributeError:
    # Even now, we may be in an interactive shell with `python -i`.
    _interactive = sys.flags.interactive

_logger = logging.getLogger('pyjedai')

if _interactive:
    _logger.setLevel(WARN)
    _logging_target = sys.stdout
else:
    _logging_target = sys.stderr

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
_logger.addHandler(_handler)

log = _logger.log
debug = _logger.debug
error = _logger.error
fatal = _logger.fatal
info = _logger.info
warning = _logger.warning
