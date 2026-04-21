# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""macOS menu bar application for TRIBE v2.

Provides a status-bar icon that manages the TRIBE v2 API server as a
background subprocess.  The server is started, stopped, and monitored
entirely from the menu bar — no terminal required.

Run directly::

    python -m tribev2.menubar

Or via the installed console script::

    tribe-menubar

Features:

- Start / Stop / Restart the FastAPI server
- Health monitoring via periodic ``/health`` polling
- Native macOS settings window (PyObjC)
- LaunchAgent integration for "Start at Login"
- macOS notifications for key lifecycle events
- Server log viewing via Console.app
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import rumps

from tribev2.config import LOG_DIR, TribeConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APP_NAME = "TRIBE v2"
_HEALTH_POLL_INTERVAL = 5  # seconds
_SHUTDOWN_TIMEOUT = 10  # seconds to wait for graceful shutdown
_STARTUP_TIMEOUT = 30  # seconds to wait for server to come up

# Status labels
_STATUS_STOPPED = "Status: Stopped  🔴"
_STATUS_STARTING = "Status: Starting  ⏳"
_STATUS_RUNNING = "Status: Running  ✅"
_STATUS_STOPPING = "Status: Stopping  ⏳"
_STATUS_ERROR = "Status: Error  ⚠️"

# SF Symbol name for the menu bar icon (macOS 11+)
# Apple's built-in brain icon — monochrome, adapts to light/dark mode
_SF_SYMBOL_NAME = "brain.head.profile"


# ---------------------------------------------------------------------------
# Server process manager
# ---------------------------------------------------------------------------


class ServerProcess:
    """Manages the uvicorn server subprocess."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._log_file = None

    @property
    def is_running(self) -> bool:
        """Check if the server process is alive."""
        if self._process is None:
            return False
        return self._process.poll() is None

    @property
    def pid(self) -> int | None:
        """Return the server process PID, or None."""
        if self._process and self._process.poll() is None:
            return self._process.pid
        return None

    def start(self, config: dict) -> bool:
        """Start the server subprocess.

        Parameters
        ----------
        config : dict
            Configuration dictionary (from TribeConfig.load()).

        Returns
        -------
        bool
            True if the process was started successfully.
        """
        if self.is_running:
            logger.warning("Server already running (PID %s)", self.pid)
            return True

        # Build environment
        env = dict(os.environ)
        env.update(TribeConfig.get_env_dict(config))

        # Ensure log directory exists
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = TribeConfig.log_path()

        # Find the Python interpreter
        python = sys.executable

        try:
            self._log_file = open(log_path, "a", encoding="utf-8")
            self._process = subprocess.Popen(
                [python, "-m", "tribev2.server"],
                env=env,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # New process group for clean shutdown
            )
            logger.info(
                "Server started (PID %s), logging to %s",
                self._process.pid, log_path,
            )
            return True
        except Exception as exc:
            logger.exception("Failed to start server: %s", exc)
            self._cleanup()
            return False

    def stop(self) -> bool:
        """Stop the server subprocess gracefully.

        Sends SIGTERM first, then SIGKILL if necessary.

        Returns
        -------
        bool
            True if the server was stopped.
        """
        if not self.is_running:
            logger.info("Server not running")
            self._cleanup()
            return True

        pid = self._process.pid
        logger.info("Stopping server (PID %s)...", pid)

        # Try SIGTERM first (graceful)
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        # Wait for graceful shutdown
        try:
            self._process.wait(timeout=_SHUTDOWN_TIMEOUT)
            logger.info("Server stopped gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("Server did not stop gracefully, sending SIGKILL")
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                self._process.wait(timeout=5)
            except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
                pass

        self._cleanup()
        return True

    def _cleanup(self) -> None:
        """Clean up process references and log file."""
        self._process = None
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None


# ---------------------------------------------------------------------------
# Health checker
# ---------------------------------------------------------------------------


class HealthChecker:
    """Polls the server's /health endpoint in a background thread."""

    def __init__(self, port: int = 8000, host: str = "127.0.0.1") -> None:
        self.port = port
        self.host = host
        self._status: str | None = None

    def check(self) -> str | None:
        """Perform a single health check.

        Returns
        -------
        str or None
            Server status string ('ok', 'idle', 'loading') or None if
            the server is unreachable.
        """
        import urllib.request
        import urllib.error
        import json

        url = f"http://{self.host}:{self.port}/health"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                self._status = data.get("status")
                return self._status
        except Exception:
            self._status = None
            return None

    @property
    def last_status(self) -> str | None:
        return self._status


# ---------------------------------------------------------------------------
# Menu bar application
# ---------------------------------------------------------------------------


class TribeMenuBarApp(rumps.App):
    """macOS menu bar application for TRIBE v2."""

    def __init__(self) -> None:
        super().__init__(
            name=_APP_NAME,
            title="T2",  # text fallback, replaced by SF Symbol below
            quit_button=None,  # We'll add our own
        )

        # Use Apple's SF Symbol for a professional menu bar icon
        self._apply_sf_symbol()

        self._config = TribeConfig.load()
        self._server = ServerProcess()
        self._health = HealthChecker(
            port=self._config.get("port", 8000),
            host="127.0.0.1",
        )
        self._settings_window = None  # reference to prevent GC

        # Build the menu
        self._status_item = rumps.MenuItem(
            _STATUS_STOPPED, callback=None
        )
        self._status_item.set_callback(None)

        self._start_item = rumps.MenuItem(
            "Start Server", callback=self._on_start
        )
        self._stop_item = rumps.MenuItem(
            "Stop Server", callback=self._on_stop
        )
        self._stop_item.set_callback(None)  # disabled initially
        self._restart_item = rumps.MenuItem(
            "Restart Server", callback=self._on_restart
        )
        self._restart_item.set_callback(None)  # disabled initially

        self._docs_item = rumps.MenuItem(
            "Open API Docs", callback=self._on_open_docs
        )
        self._log_item = rumps.MenuItem(
            "View Server Log", callback=self._on_view_log
        )
        self._settings_item = rumps.MenuItem(
            "Settings…", callback=self._on_settings
        )
        self._quit_item = rumps.MenuItem(
            "Quit TRIBE v2", callback=self._on_quit
        )

        self.menu = [
            self._status_item,
            None,  # separator
            self._start_item,
            self._stop_item,
            self._restart_item,
            None,  # separator
            self._docs_item,
            self._log_item,
            None,  # separator
            self._settings_item,
            None,  # separator
            self._quit_item,
        ]

        # Start the health check timer
        self._timer = rumps.Timer(self._health_poll, _HEALTH_POLL_INTERVAL)
        self._timer.start()

    # ── Icon setup ───────────────────────────────────────────────────

    def _apply_sf_symbol(self) -> None:
        """Load an SF Symbol as the menu bar icon via AppKit.

        Uses Apple's built-in 'brain.head.profile' symbol which is
        professionally designed, monochrome, and automatically adapts
        to light/dark menu bars. Falls back to text title on older macOS.
        """
        try:
            from AppKit import NSImage

            image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                _SF_SYMBOL_NAME, "TRIBE v2"
            )
            if image is not None:
                image.setTemplate_(True)
                # Store the NSImage — we'll apply it once the run loop starts
                self._sf_icon = image
                logger.info("SF Symbol '%s' loaded", _SF_SYMBOL_NAME)
            else:
                self._sf_icon = None
                logger.info(
                    "SF Symbol '%s' not available, using text fallback",
                    _SF_SYMBOL_NAME,
                )
        except ImportError:
            self._sf_icon = None
            logger.info("AppKit not available, using text fallback")

    def _apply_icon_to_button(self) -> None:
        """Apply the SF Symbol to the NSStatusItem button.

        Must be called after rumps has created the status item (i.e.,
        inside the run loop). Called by the first health poll tick.
        """
        if not hasattr(self, '_sf_icon') or self._sf_icon is None:
            return
        if hasattr(self, '_icon_applied') and self._icon_applied:
            return

        try:
            # rumps stores the NSStatusItem at _nsapp.nsstatusitem
            nsapp = getattr(self, '_nsapp', None)
            if nsapp is None:
                return
            status_item = getattr(nsapp, 'nsstatusitem', None)
            if status_item is not None:
                button = status_item.button()
                if button is not None:
                    button.setImage_(self._sf_icon)
                    button.setTitle_("")  # remove text
                    self._icon_applied = True
                    logger.info("SF Symbol applied to menu bar")
        except Exception as exc:
            logger.debug("Could not apply SF Symbol: %s", exc)

    # ── Status update ────────────────────────────────────────────────

    def _update_status(self, status_text: str) -> None:
        """Update the status menu item text."""
        self._status_item.title = status_text

    def _set_running_state(self) -> None:
        """Update UI to show server is running."""
        self._update_status(_STATUS_RUNNING)
        self._start_item.set_callback(None)
        self._stop_item.set_callback(self._on_stop)
        self._restart_item.set_callback(self._on_restart)

    def _set_stopped_state(self) -> None:
        """Update UI to show server is stopped."""
        self._update_status(_STATUS_STOPPED)
        self._start_item.set_callback(self._on_start)
        self._stop_item.set_callback(None)
        self._restart_item.set_callback(None)

    def _set_starting_state(self) -> None:
        """Update UI to show server is starting."""
        self._update_status(_STATUS_STARTING)
        self._start_item.set_callback(None)
        self._stop_item.set_callback(None)
        self._restart_item.set_callback(None)

    def _set_stopping_state(self) -> None:
        """Update UI to show server is stopping."""
        self._update_status(_STATUS_STOPPING)
        self._start_item.set_callback(None)
        self._stop_item.set_callback(None)
        self._restart_item.set_callback(None)

    # ── Health polling ───────────────────────────────────────────────

    def _health_poll(self, _timer=None) -> None:
        """Periodic health check (called by rumps.Timer)."""
        # Apply the SF Symbol icon once the run loop is active
        self._apply_icon_to_button()

        if not self._server.is_running:
            self._set_stopped_state()
            return

        status = self._health.check()
        if status in ("ok", "idle", "loading"):
            self._set_running_state()
        else:
            # Process alive but not responding — probably still starting
            if self._status_item.title != _STATUS_STARTING:
                self._set_starting_state()

    # ── Menu callbacks ───────────────────────────────────────────────

    def _on_start(self, sender) -> None:
        """Start the server in a background thread."""
        self._config = TribeConfig.load()
        self._health.port = self._config.get("port", 8000)
        self._set_starting_state()

        def _start_bg():
            success = self._server.start(self._config)
            if success:
                # Wait for the server to respond
                for _ in range(int(_STARTUP_TIMEOUT / 2)):
                    time.sleep(2)
                    status = self._health.check()
                    if status:
                        rumps.notification(
                            title="TRIBE v2",
                            subtitle="Server Started",
                            message=f"Running on port {self._config.get('port', 8000)}",
                        )
                        return
                # Server started but health check didn't respond yet
                # The periodic timer will pick it up
                rumps.notification(
                    title="TRIBE v2",
                    subtitle="Server Starting",
                    message="Model loading may take a moment...",
                )
            else:
                rumps.notification(
                    title="TRIBE v2",
                    subtitle="Server Failed",
                    message="Check the log for details.",
                )

        thread = threading.Thread(target=_start_bg, daemon=True)
        thread.start()

    def _on_stop(self, sender) -> None:
        """Stop the server."""
        self._set_stopping_state()

        def _stop_bg():
            self._server.stop()
            rumps.notification(
                title="TRIBE v2",
                subtitle="Server Stopped",
                message="The server has been shut down.",
            )

        thread = threading.Thread(target=_stop_bg, daemon=True)
        thread.start()

    def _on_restart(self, sender) -> None:
        """Restart the server."""
        self._set_stopping_state()

        def _restart_bg():
            self._server.stop()
            time.sleep(1)
            self._config = TribeConfig.load()
            self._health.port = self._config.get("port", 8000)
            self._server.start(self._config)
            rumps.notification(
                title="TRIBE v2",
                subtitle="Server Restarting",
                message="The server is restarting...",
            )

        thread = threading.Thread(target=_restart_bg, daemon=True)
        thread.start()

    def _on_open_docs(self, sender) -> None:
        """Open the API docs in the default browser."""
        port = self._config.get("port", 8000)
        webbrowser.open(f"http://localhost:{port}/docs")

    def _on_view_log(self, sender) -> None:
        """Open the server log in Console.app."""
        log_path = TribeConfig.log_path()
        if log_path.is_file():
            subprocess.Popen(["open", "-a", "Console", str(log_path)])
        else:
            rumps.notification(
                title="TRIBE v2",
                subtitle="No Log File",
                message=f"Log file not found at {log_path}",
            )

    def _on_settings(self, sender) -> None:
        """Open the settings window."""
        try:
            from tribev2.settings_gui import show_settings_window

            self._settings_window = show_settings_window(
                on_save=self._on_settings_saved
            )
        except ImportError:
            rumps.notification(
                title="TRIBE v2",
                subtitle="Settings Unavailable",
                message="Install with: pip install 'tribev2[menubar]'",
            )
        except Exception as exc:
            logger.exception("Failed to open settings: %s", exc)
            rumps.notification(
                title="TRIBE v2",
                subtitle="Settings Error",
                message=str(exc),
            )

    def _on_settings_saved(self, config: dict, needs_restart: bool) -> None:
        """Callback after settings are saved."""
        self._config = config
        self._health.port = config.get("port", 8000)

        if needs_restart and self._server.is_running:
            response = rumps.alert(
                title="Restart Required",
                message=(
                    "Server settings have changed. "
                    "Would you like to restart the server now?"
                ),
                ok="Restart Now",
                cancel="Later",
            )
            if response == 1:  # OK button
                self._on_restart(None)

    def _on_quit(self, sender) -> None:
        """Quit the menu bar app (optionally stop the server)."""
        if self._server.is_running:
            response = rumps.alert(
                title="Quit TRIBE v2",
                message=(
                    "The server is still running. "
                    "Would you like to stop it before quitting?"
                ),
                ok="Stop & Quit",
                cancel="Quit Only",
            )
            if response == 1:  # OK = Stop & Quit
                self._server.stop()

        if self._timer:
            self._timer.stop()

        rumps.quit_application()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the TRIBE v2 menu bar application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Starting TRIBE v2 menu bar app")
    app = TribeMenuBarApp()
    app.run()


if __name__ == "__main__":
    main()
