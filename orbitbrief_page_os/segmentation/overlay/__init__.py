"""Overlay rendering split by concern (colours, blue shells, title washes, …).

Detection still lives in ``detect_standalone``; this package holds **paint-only**
helpers so outline vs title-highlight logic can evolve without one giant
``render_overlay`` closure.
"""

from __future__ import annotations
