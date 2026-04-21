# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent configuration manager for TRIBE v2.

Stores user preferences in ``~/Library/Application Support/TribeV2/config.json``
and exposes helpers to convert them to the ``TRIBE_*`` environment variables
expected by :mod:`tribev2.server`.

Usage::

    from tribev2.config import TribeConfig

    cfg = TribeConfig.load()
    print(cfg)       # {'host': '0.0.0.0', 'port': 8000, ...}

    cfg['port'] = 9000
    TribeConfig.save(cfg)

    env = TribeConfig.get_env_dict(cfg)
    # {'TRIBE_HOST': '0.0.0.0', 'TRIBE_PORT': '9000', ...}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — follow macOS conventions
# ---------------------------------------------------------------------------

_APP_SUPPORT = Path.home() / "Library" / "Application Support" / "TribeV2"
_LOG_DIR = Path.home() / "Library" / "Logs" / "TribeV2"

CONFIG_DIR: Path = _APP_SUPPORT
CONFIG_FILE: Path = _APP_SUPPORT / "config.json"
LOG_DIR: Path = _LOG_DIR
LAUNCHD_LABEL: str = "com.meta.tribev2.menubar"

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 8000,
    "device": "auto",
    "idle_timeout": 120,
    "cache_dir": "./cache",
    "stimuli_dir": "./stimuli",
    "model_id": "facebook/tribev2",
    "start_at_login": False,
    "log_level": "INFO",
}

# Map from config key → environment variable name
_ENV_MAP: dict[str, str] = {
    "host": "TRIBE_HOST",
    "port": "TRIBE_PORT",
    "device": "TRIBE_DEVICE",
    "idle_timeout": "TRIBE_IDLE_TIMEOUT_SECONDS",
    "cache_dir": "TRIBE_CACHE_DIR",
    "stimuli_dir": "TRIBE_STIMULI_DIR",
    "model_id": "TRIBE_MODEL_ID",
}

# Human-readable metadata for each setting (used by the settings GUI)
SETTING_META: dict[str, dict[str, Any]] = {
    "host": {
        "label": "Bind Address",
        "description": "Network interface to bind to. Use 0.0.0.0 for all interfaces, 127.0.0.1 for local only.",
        "type": "text",
        "group": "server",
    },
    "port": {
        "label": "Port",
        "description": "HTTP port for the API server.",
        "type": "number",
        "min": 1024,
        "max": 65535,
        "group": "server",
    },
    "device": {
        "label": "Compute Device",
        "description": "Torch device for inference. 'auto' selects MPS on Apple Silicon.",
        "type": "select",
        "options": ["auto", "mps", "cpu"],
        "group": "model",
    },
    "idle_timeout": {
        "label": "Idle Timeout (seconds)",
        "description": "Seconds of inactivity before the model is unloaded to free memory. Set to 0 to disable.",
        "type": "number",
        "min": 0,
        "max": 3600,
        "group": "model",
    },
    "cache_dir": {
        "label": "Cache Directory",
        "description": "Directory for caching extracted features. Safe to delete; rebuilds on next run.",
        "type": "text",
        "group": "storage",
    },
    "stimuli_dir": {
        "label": "Stimuli Directory",
        "description": "Directory containing reference .txt files for emotional state classification.",
        "type": "text",
        "group": "storage",
    },
    "model_id": {
        "label": "Model ID",
        "description": "HuggingFace repository or local path for the TRIBE v2 checkpoint.",
        "type": "text",
        "group": "model",
    },
    "start_at_login": {
        "label": "Start at Login",
        "description": "Automatically launch the TRIBE v2 menu bar app when you log in.",
        "type": "toggle",
        "group": "system",
    },
    "log_level": {
        "label": "Log Level",
        "description": "Logging verbosity for the server process.",
        "type": "select",
        "options": ["DEBUG", "INFO", "WARNING", "ERROR"],
        "group": "system",
    },
}


# ---------------------------------------------------------------------------
# TribeConfig — public API
# ---------------------------------------------------------------------------


class TribeConfig:
    """Manages persistent configuration for TRIBE v2.

    All methods are classmethods/staticmethods — there is no need to
    instantiate this class.  Configuration is stored as a plain JSON
    file and all values are primitive types (str, int, bool).
    """

    @staticmethod
    def load() -> dict[str, Any]:
        """Load configuration from disk, filling in defaults for missing keys.

        Creates the config directory and a default config file if they
        don't exist yet.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary with all keys from :data:`DEFAULTS`.
        """
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        if CONFIG_FILE.is_file():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    stored = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Corrupt config file, resetting to defaults: %s", exc)
                stored = {}
        else:
            stored = {}

        # Merge with defaults (stored values take precedence)
        config = {**DEFAULTS, **stored}

        # Write back to ensure new defaults are persisted
        if config != stored:
            TribeConfig.save(config)

        return config

    @staticmethod
    def save(config: dict[str, Any]) -> None:
        """Persist *config* to disk as JSON.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary. Unknown keys are preserved but
            won't be used by the application.
        """
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info("Configuration saved to %s", CONFIG_FILE)

    @staticmethod
    def get_env_dict(config: dict[str, Any] | None = None) -> dict[str, str]:
        """Convert configuration to ``TRIBE_*`` environment variables.

        Parameters
        ----------
        config : dict or None
            If ``None``, the saved config is loaded from disk.

        Returns
        -------
        dict[str, str]
            Environment variable mapping ready to pass to
            ``subprocess.Popen(env=…)`` or ``os.environ.update(…)``.
        """
        if config is None:
            config = TribeConfig.load()

        env: dict[str, str] = {}
        for key, env_var in _ENV_MAP.items():
            if key in config:
                env[env_var] = str(config[key])
        return env

    @staticmethod
    def log_path() -> Path:
        """Return the path to the current server log file.

        Creates the log directory if it doesn't exist.
        """
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        return LOG_DIR / "server.log"

    @staticmethod
    def reset() -> dict[str, Any]:
        """Reset configuration to defaults and persist.

        Returns
        -------
        dict[str, Any]
            The default configuration.
        """
        TribeConfig.save(dict(DEFAULTS))
        logger.info("Configuration reset to defaults")
        return dict(DEFAULTS)
