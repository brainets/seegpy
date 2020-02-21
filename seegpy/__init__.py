"""
Seegpy
======

Utility functions for analyzing sEEG data
"""
import logging

from seegpy import contacts, io, labelling, mesh, pipeline, transform, utils  # noqa

__version__ = "0.0.0"

# -----------------------------------------------------------------------------
# Set 'info' as the default logging level
logger = logging.getLogger('seegpy')
io.set_log_level('info')
