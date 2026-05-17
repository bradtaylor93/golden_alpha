"""Repository-root import shim for local development.

The implementation package lives in ``ath_reversion_research/`` inside this
folder.  This shim keeps ``import ath_reversion_research`` working when tests
are launched from the parent repository root.
"""

from .ath_reversion_research import *  # noqa: F401,F403
