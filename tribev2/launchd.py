# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Manage a macOS LaunchAgent for the TRIBE v2 menu bar app.

This module creates, installs, and removes a ``launchd`` plist so that
the menu bar application can optionally start at login.

Usage::

    from tribev2.launchd import LaunchdManager

    mgr = LaunchdManager()
    mgr.install()       # register the LaunchAgent
    mgr.uninstall()     # remove the LaunchAgent
    mgr.is_installed()  # check registration state
"""

from __future__ import annotations

import logging
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAUNCHD_LABEL = "com.meta.tribev2.menubar"
_LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
_PLIST_FILENAME = f"{LAUNCHD_LABEL}.plist"


def plist_path() -> Path:
    """Return the path where the LaunchAgent plist will be installed."""
    return _LAUNCH_AGENTS_DIR / _PLIST_FILENAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_console_script() -> str:
    """Find the absolute path to the ``tribe-menubar`` console script.

    Searches in order:
    1. ``sys.prefix + /bin/tribe-menubar``  (venv / pip install)
    2. ``shutil.which("tribe-menubar")``    (PATH lookup)
    3. Falls back to invoking via ``python -m tribev2.menubar``
    """
    # Check within the active virtual-env / sys.prefix
    candidate = Path(sys.prefix) / "bin" / "tribe-menubar"
    if candidate.is_file():
        return str(candidate)

    # PATH lookup
    found = shutil.which("tribe-menubar")
    if found:
        return found

    # Fallback — use the Python interpreter + module invocation
    return ""


def _build_plist() -> dict:
    """Build the LaunchAgent property-list dictionary.

    Returns
    -------
    dict
        A dictionary suitable for ``plistlib.dump()``.
    """
    script = _find_console_script()

    if script:
        program_args = [script]
    else:
        # Fallback: run as a Python module with the current interpreter
        python = sys.executable or shutil.which("python3") or "/usr/bin/python3"
        program_args = [python, "-m", "tribev2.menubar"]

    from tribev2.config import LOG_DIR

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": program_args,
        "RunAtLoad": True,
        "KeepAlive": False,  # Don't auto-restart — user controls via menu
        "StandardOutPath": str(LOG_DIR / "menubar.log"),
        "StandardErrorPath": str(LOG_DIR / "menubar.err"),
        "ProcessType": "Interactive",
    }


# ---------------------------------------------------------------------------
# LaunchdManager
# ---------------------------------------------------------------------------


class LaunchdManager:
    """Manage a macOS LaunchAgent for the TRIBE v2 menu bar app.

    All methods are safe to call repeatedly (idempotent).
    """

    @staticmethod
    def install() -> bool:
        """Write the plist and register the LaunchAgent.

        Returns
        -------
        bool
            ``True`` if the agent was successfully registered.
        """
        path = plist_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        plist = _build_plist()

        # Write the plist file
        with open(path, "wb") as f:
            plistlib.dump(plist, f)

        logger.info("Wrote LaunchAgent plist to %s", path)

        # Register with launchd
        uid = os.getuid()
        try:
            subprocess.run(
                ["launchctl", "bootstrap", f"gui/{uid}", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("LaunchAgent registered: %s", LAUNCHD_LABEL)
            return True
        except subprocess.CalledProcessError as exc:
            # Error 37 = "already loaded" — that's fine
            if "37" in (exc.stderr or ""):
                logger.info("LaunchAgent already registered")
                return True
            logger.warning(
                "Failed to register LaunchAgent: %s", exc.stderr or exc
            )
            return False

    @staticmethod
    def uninstall() -> bool:
        """Unregister the LaunchAgent and remove the plist file.

        Returns
        -------
        bool
            ``True`` if the agent was successfully unregistered.
        """
        path = plist_path()
        uid = os.getuid()

        # Unregister from launchd
        try:
            subprocess.run(
                ["launchctl", "bootout", f"gui/{uid}/{LAUNCHD_LABEL}"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("LaunchAgent unregistered: %s", LAUNCHD_LABEL)
        except subprocess.CalledProcessError as exc:
            # "Could not find" = already unloaded, that's fine
            stderr = exc.stderr or ""
            if "could not find" in stderr.lower() or "113" in stderr:
                logger.info("LaunchAgent was not registered, nothing to remove")
            else:
                logger.warning("Failed to unregister LaunchAgent: %s", stderr)

        # Remove the plist file
        if path.is_file():
            path.unlink()
            logger.info("Removed plist file: %s", path)

        return True

    @staticmethod
    def is_installed() -> bool:
        """Check whether the LaunchAgent is currently registered.

        Returns
        -------
        bool
            ``True`` if the agent is loaded in ``launchctl``.
        """
        try:
            result = subprocess.run(
                ["launchctl", "list"],
                capture_output=True,
                text=True,
            )
            return LAUNCHD_LABEL in (result.stdout or "")
        except subprocess.CalledProcessError:
            return False

    @staticmethod
    def plist_exists() -> bool:
        """Check whether the plist file exists on disk."""
        return plist_path().is_file()
