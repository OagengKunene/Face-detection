"""Native widget smoke tests run on Windows CI or a machine with an X display."""

import os
import tkinter as tk
from tkinter import ttk

import pytest

from bioguard.gui import Application

pytestmark = pytest.mark.skipif(
    os.name != "nt" and not os.environ.get("DISPLAY"), reason="A desktop display is required."
)


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def test_create_vault_unlock_workspace_and_lock(tmp_path):
    root = tk.Tk()
    try:
        app = Application(root, tmp_path / "data", tmp_path / "models")
        root.update()
        entries = [w for w in descendants(root) if isinstance(w, ttk.Entry)]
        for entry in entries:
            entry.insert(0, "a testing vault passphrase")
        buttons = [w for w in descendants(root) if isinstance(w, ttk.Button)]
        next(b for b in buttons if b["text"] == "Create vault").invoke()
        root.update()
        assert app.vault is not None
        assert app.person_id.winfo_exists()
        app.begin_enrol()
        assert app.status.get() == "Start the camera first."
        app.lock()
        root.update()
        assert app.vault is None
        assert (tmp_path / "data" / "vault.sqlite3").exists()
    finally:
        app.close()
