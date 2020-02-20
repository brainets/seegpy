"""
Seegpy
======

Utility functions for analyzing sEEG data
"""
import logging

from seegpy import io, labelling, mesh, transform, utils  # noqa

__version__ = "0.0.0"

# -----------------------------------------------------------------------------
# Set 'info' as the default logging level
logger = logging.getLogger('seegpy')
io.set_log_level('info')
