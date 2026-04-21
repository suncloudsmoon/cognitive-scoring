# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Native macOS settings window for the TRIBE v2 menu bar app.

Uses PyObjC (AppKit) to build a tabbed preferences window with native
macOS controls.  Each setting from :data:`tribev2.config.SETTING_META`
is rendered as the appropriate control type (text field, dropdown,
toggle switch, number stepper).

This module is imported lazily by :mod:`tribev2.menubar` when the user
clicks "Settings…" — it is never imported at startup to keep launch fast.

Usage::

    from tribev2.settings_gui import show_settings_window

    # Call from a rumps callback — opens the window modally
    show_settings_window(on_save_callback=my_restart_fn)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Guard the entire module — these are macOS-only imports
try:
    import AppKit
    import objc
    from Foundation import (
        NSMakeRect,
        NSObject,
    )
except ImportError as exc:
    raise ImportError(
        "PyObjC is required for the settings GUI. "
        "Install with: pip install 'tribev2[menubar]'"
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW_WIDTH = 520
_WINDOW_HEIGHT = 440
_PADDING = 20
_LABEL_HEIGHT = 17
_CONTROL_HEIGHT = 24
_ROW_SPACING = 52
_DESCRIPTION_HEIGHT = 14

# Tab groups in display order
_TAB_ORDER = [
    ("server", "Server"),
    ("model", "Model"),
    ("storage", "Storage"),
    ("system", "System"),
]

# ---------------------------------------------------------------------------
# Settings window delegate
# ---------------------------------------------------------------------------


class _SettingsWindowDelegate(NSObject):
    """Delegate that handles settings window close."""

    def windowWillClose_(self, notification):
        """Release the reference when the window closes."""
        # Allow garbage collection
        pass


# ---------------------------------------------------------------------------
# Settings window builder
# ---------------------------------------------------------------------------


def _create_label(text: str, frame: tuple, font_size: float = 13.0,
                  bold: bool = False) -> AppKit.NSTextField:
    """Create a static label (non-editable text field)."""
    label = AppKit.NSTextField.alloc().initWithFrame_(NSMakeRect(*frame))
    label.setStringValue_(text)
    label.setBezeled_(False)
    label.setDrawsBackground_(False)
    label.setEditable_(False)
    label.setSelectable_(False)
    if bold:
        label.setFont_(AppKit.NSFont.boldSystemFontOfSize_(font_size))
    else:
        label.setFont_(AppKit.NSFont.systemFontOfSize_(font_size))
    return label


def _create_description(text: str, frame: tuple) -> AppKit.NSTextField:
    """Create a small description label."""
    desc = _create_label(text, frame, font_size=11.0)
    desc.setTextColor_(AppKit.NSColor.secondaryLabelColor())
    return desc


def _create_text_field(value: str, frame: tuple) -> AppKit.NSTextField:
    """Create an editable text field."""
    field = AppKit.NSTextField.alloc().initWithFrame_(NSMakeRect(*frame))
    field.setStringValue_(str(value))
    field.setFont_(AppKit.NSFont.systemFontOfSize_(13.0))
    field.setBordered_(True)
    field.setBezeled_(True)
    field.setBezelStyle_(AppKit.NSTextFieldRoundedBezel)
    return field


def _create_number_field(value: int | float, frame: tuple,
                         min_val: int = 0,
                         max_val: int = 65535) -> AppKit.NSTextField:
    """Create a number-only text field."""
    field = _create_text_field(str(int(value)), frame)

    formatter = AppKit.NSNumberFormatter.alloc().init()
    formatter.setNumberStyle_(AppKit.NSNumberFormatterNoStyle)
    formatter.setMinimum_(AppKit.NSDecimalNumber.numberWithInt_(min_val))
    formatter.setMaximum_(AppKit.NSDecimalNumber.numberWithInt_(max_val))
    field.setFormatter_(formatter)
    return field


def _create_dropdown(options: list[str], current: str,
                     frame: tuple) -> AppKit.NSPopUpButton:
    """Create a dropdown (popup button)."""
    popup = AppKit.NSPopUpButton.alloc().initWithFrame_pullsDown_(
        NSMakeRect(*frame), False
    )
    popup.removeAllItems()
    popup.addItemsWithTitles_(options)
    if current in options:
        popup.selectItemWithTitle_(current)
    popup.setFont_(AppKit.NSFont.systemFontOfSize_(13.0))
    return popup


def _create_toggle(value: bool, frame: tuple) -> AppKit.NSButton:
    """Create a toggle switch (checkbox-style)."""
    toggle = AppKit.NSButton.alloc().initWithFrame_(NSMakeRect(*frame))
    toggle.setButtonType_(AppKit.NSSwitchButton)
    toggle.setTitle_("")
    toggle.setState_(AppKit.NSControlStateValueOn if value else AppKit.NSControlStateValueOff)
    return toggle


class SettingsWindow:
    """Builds and manages the native macOS settings window."""

    def __init__(
        self,
        on_save: Callable[[dict[str, Any], bool], None] | None = None,
    ) -> None:
        self._on_save = on_save
        self._controls: dict[str, Any] = {}
        self._window: AppKit.NSWindow | None = None
        self._original_config: dict[str, Any] = {}

    def show(self) -> None:
        """Build and display the settings window."""
        from tribev2.config import SETTING_META, TribeConfig

        self._original_config = TribeConfig.load()
        config = dict(self._original_config)

        # --- Window ---
        style = (
            AppKit.NSWindowStyleMaskTitled
            | AppKit.NSWindowStyleMaskClosable
            | AppKit.NSWindowStyleMaskMiniaturizable
        )
        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(200, 200, _WINDOW_WIDTH, _WINDOW_HEIGHT),
            style,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        window.setTitle_("TRIBE v2 Settings")
        window.center()
        window.setLevel_(AppKit.NSFloatingWindowLevel)

        # Delegate to handle close
        delegate = _SettingsWindowDelegate.alloc().init()
        window.setDelegate_(delegate)
        self._delegate = delegate  # prevent GC

        content = window.contentView()

        # --- Tab view ---
        tab_view = AppKit.NSTabView.alloc().initWithFrame_(
            NSMakeRect(
                _PADDING,
                60,  # leave room for buttons at bottom
                _WINDOW_WIDTH - 2 * _PADDING,
                _WINDOW_HEIGHT - 80,
            )
        )

        for group_id, group_label in _TAB_ORDER:
            tab_item = AppKit.NSTabViewItem.alloc().initWithIdentifier_(group_id)
            tab_item.setLabel_(group_label)

            tab_content = AppKit.NSView.alloc().initWithFrame_(
                NSMakeRect(0, 0, _WINDOW_WIDTH - 2 * _PADDING - 20,
                           _WINDOW_HEIGHT - 140)
            )

            # Collect settings for this group
            group_settings = [
                (key, meta)
                for key, meta in SETTING_META.items()
                if meta.get("group") == group_id
            ]

            tab_height = _WINDOW_HEIGHT - 140
            y = tab_height - _PADDING - 10

            for key, meta in group_settings:
                value = config.get(key, "")
                label_text = meta.get("label", key)
                description = meta.get("description", "")
                control_type = meta.get("type", "text")

                content_width = _WINDOW_WIDTH - 2 * _PADDING - 40

                # Label
                label = _create_label(
                    f"{label_text}:",
                    (_PADDING, y, content_width, _LABEL_HEIGHT),
                    bold=True,
                )
                tab_content.addSubview_(label)
                y -= _CONTROL_HEIGHT + 4

                # Control
                control_frame = (_PADDING, y, content_width - 10, _CONTROL_HEIGHT)

                if control_type == "text":
                    control = _create_text_field(str(value), control_frame)
                elif control_type == "number":
                    control = _create_number_field(
                        value, control_frame,
                        min_val=meta.get("min", 0),
                        max_val=meta.get("max", 65535),
                    )
                elif control_type == "select":
                    options = meta.get("options", [])
                    control = _create_dropdown(
                        options, str(value), control_frame
                    )
                elif control_type == "toggle":
                    control = _create_toggle(bool(value), control_frame)
                else:
                    control = _create_text_field(str(value), control_frame)

                tab_content.addSubview_(control)
                self._controls[key] = (control, control_type)
                y -= _DESCRIPTION_HEIGHT + 6

                # Description
                if description:
                    desc = _create_description(
                        description,
                        (_PADDING, y, content_width, _DESCRIPTION_HEIGHT + 4),
                    )
                    tab_content.addSubview_(desc)
                    y -= _ROW_SPACING - _CONTROL_HEIGHT

            tab_item.setView_(tab_content)
            tab_view.addTabViewItem_(tab_item)

        content.addSubview_(tab_view)

        # --- Bottom buttons ---
        btn_y = 16
        btn_height = 32

        # Reset to Defaults
        reset_btn = AppKit.NSButton.alloc().initWithFrame_(
            NSMakeRect(_PADDING, btn_y, 140, btn_height)
        )
        reset_btn.setTitle_("Reset to Defaults")
        reset_btn.setBezelStyle_(AppKit.NSBezelStyleRounded)
        reset_btn.setTarget_(self)
        reset_btn.setAction_(objc.selector(self._on_reset_, signature=b"v@:@"))
        content.addSubview_(reset_btn)

        # Cancel
        cancel_btn = AppKit.NSButton.alloc().initWithFrame_(
            NSMakeRect(_WINDOW_WIDTH - _PADDING - 85 - 10 - 85, btn_y, 85, btn_height)
        )
        cancel_btn.setTitle_("Cancel")
        cancel_btn.setBezelStyle_(AppKit.NSBezelStyleRounded)
        cancel_btn.setKeyEquivalent_("\x1b")  # Escape key
        cancel_btn.setTarget_(self)
        cancel_btn.setAction_(objc.selector(self._on_cancel_, signature=b"v@:@"))
        content.addSubview_(cancel_btn)

        # Save
        save_btn = AppKit.NSButton.alloc().initWithFrame_(
            NSMakeRect(_WINDOW_WIDTH - _PADDING - 85, btn_y, 85, btn_height)
        )
        save_btn.setTitle_("Save")
        save_btn.setBezelStyle_(AppKit.NSBezelStyleRounded)
        save_btn.setKeyEquivalent_("\r")  # Enter key
        save_btn.setTarget_(self)
        save_btn.setAction_(objc.selector(self._on_save_, signature=b"v@:@"))
        content.addSubview_(save_btn)

        self._window = window
        window.makeKeyAndOrderFront_(None)

        # Bring our app to the front
        AppKit.NSApp.activateIgnoringOtherApps_(True)

    def _read_controls(self) -> dict[str, Any]:
        """Read current values from all controls."""
        from tribev2.config import DEFAULTS

        values: dict[str, Any] = {}
        for key, (control, control_type) in self._controls.items():
            if control_type == "text":
                values[key] = control.stringValue()
            elif control_type == "number":
                values[key] = control.intValue()
            elif control_type == "select":
                values[key] = control.titleOfSelectedItem()
            elif control_type == "toggle":
                values[key] = control.state() == AppKit.NSControlStateValueOn
            else:
                values[key] = control.stringValue()

        # Ensure proper types
        for key, val in values.items():
            default = DEFAULTS.get(key)
            if isinstance(default, int) and not isinstance(default, bool):
                try:
                    values[key] = int(val)
                except (ValueError, TypeError):
                    values[key] = default
            elif isinstance(default, bool):
                values[key] = bool(val)

        return values

    @objc.python_method
    def _close_window(self):
        """Close the settings window."""
        if self._window:
            self._window.close()
            self._window = None

    def _on_save_(self, sender):
        """Handle Save button click."""
        from tribev2.config import TribeConfig

        values = self._read_controls()
        config = {**self._original_config, **values}
        TribeConfig.save(config)
        logger.info("Settings saved")

        # Check if server-affecting settings changed
        server_keys = {"host", "port", "device", "idle_timeout", "model_id",
                       "cache_dir", "stimuli_dir"}
        needs_restart = any(
            config.get(k) != self._original_config.get(k)
            for k in server_keys
        )

        # Handle start_at_login toggle
        login_changed = config.get("start_at_login") != self._original_config.get("start_at_login")
        if login_changed:
            try:
                from tribev2.launchd import LaunchdManager
                if config.get("start_at_login"):
                    LaunchdManager.install()
                else:
                    LaunchdManager.uninstall()
            except Exception as exc:
                logger.warning("Failed to update login item: %s", exc)

        self._close_window()

        if self._on_save:
            self._on_save(config, needs_restart)

    def _on_cancel_(self, sender):
        """Handle Cancel button click."""
        self._close_window()

    def _on_reset_(self, sender):
        """Handle Reset to Defaults button click."""
        from tribev2.config import DEFAULTS, SETTING_META

        for key, (control, control_type) in self._controls.items():
            default = DEFAULTS.get(key, "")
            if control_type == "text":
                control.setStringValue_(str(default))
            elif control_type == "number":
                control.setIntValue_(int(default))
            elif control_type == "select":
                options = SETTING_META.get(key, {}).get("options", [])
                if str(default) in options:
                    control.selectItemWithTitle_(str(default))
            elif control_type == "toggle":
                state = (
                    AppKit.NSControlStateValueOn
                    if default
                    else AppKit.NSControlStateValueOff
                )
                control.setState_(state)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def show_settings_window(
    on_save: Callable[[dict[str, Any], bool], None] | None = None,
) -> SettingsWindow:
    """Open the settings window.

    Parameters
    ----------
    on_save : callable, optional
        Called after the user saves settings.  Receives ``(config, needs_restart)``
        where ``config`` is the new configuration dict, and ``needs_restart``
        is ``True`` if server-affecting settings changed.

    Returns
    -------
    SettingsWindow
        The window instance (keep a reference to prevent GC).
    """
    win = SettingsWindow(on_save=on_save)
    win.show()
    return win
