"""Native Tk desktop interface. Camera frames and face samples stay in memory."""

import csv
import queue
import sqlite3
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageOps, ImageTk

from bioguard import __version__
from bioguard.camera import Camera, CameraError
from bioguard.config import CAPTURE_TIMEOUT, ENROLMENT_SAMPLES, SAMPLE_INTERVAL, THRESHOLD
from bioguard.decisions import MatchWindow, cosine, make_template, validate_id
from bioguard.models import ensure_models
from bioguard.storage import Vault, VaultError
from bioguard.vision import FaceEngine

BG = "#0b1220"
PANEL = "#142033"
TEXT = "#edf2f7"
MUTED = "#a8b8cb"
ACCENT = "#66dfca"
WARNING = "#ffd080"


class Application:
    def __init__(self, root: tk.Tk, directory: Path, models: Path, camera_index: int = 0):
        self.root = root
        self.directory = directory
        self.models = models
        self.vault: Vault | None = None
        self.engine: FaceEngine | None = None
        self.camera: Camera | None = None
        self.camera_index = tk.IntVar(value=camera_index)
        self.mode: str | None = None
        self.samples = []
        self.window: MatchWindow | None = None
        self.target = None
        self.last_score = None
        self.last_sample = 0.0
        self.started = 0.0
        self.last_inference = 0.0
        self.analysis = None
        self.messages = queue.Queue()
        self.closed = False
        self.downloading = False
        self.tick_job = None
        root.title(f"Face-detection {__version__} | Local biometric workspace")
        root.geometry("1160x850")
        root.minsize(1030, 760)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._styles()
        self._unlock_screen()
        self.root.after(100, self._poll_messages)

    def _styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 11))
        style.configure("Card.TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", font=("Segoe UI", 26, "bold"), foreground=ACCENT)
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        style.configure("Primary.TButton", background=ACCENT, foreground=BG)
        style.map("Primary.TButton", background=[("active", "#9bf0e1")])
        style.configure("TEntry", padding=7, font=("Segoe UI", 11))
        style.configure("TCheckbutton", background=BG, foreground=TEXT, padding=5)
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 8), font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=26, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=PANEL)

    def _clear(self):
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.unbind(event)
        for child in self.root.winfo_children():
            child.destroy()

    def _unlock_screen(self):
        self._clear()
        screen = ttk.Frame(self.root, padding=50)
        screen.pack(expand=True)
        ttk.Label(screen, text="FACE DETECTION", style="Title.TLabel").pack(anchor="w")
        ttk.Label(screen, text="Your face. Your PC. Your data.", font=("Segoe UI", 16)).pack(
            anchor="w", pady=(8, 24)
        )
        exists = (self.directory / "vault.sqlite3").exists()
        title = "Unlock your biometric vault" if exists else "Create your biometric vault"
        ttk.Label(screen, text=title, font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(
            screen,
            text="Your passphrase encrypts face templates and activity records.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(8, 16))
        ttk.Label(screen, text="Vault passphrase").pack(anchor="w")
        password = ttk.Entry(screen, show="*", width=44)
        password.pack(fill="x", pady=6)
        confirmation = None
        if not exists:
            ttk.Label(screen, text="Confirm passphrase (at least 12 characters)").pack(anchor="w")
            confirmation = ttk.Entry(screen, show="*", width=44)
            confirmation.pack(fill="x", pady=6)
        feedback = ttk.Label(screen, text="", foreground=WARNING, wraplength=540)
        feedback.pack(anchor="w", pady=8)

        def unlock():
            if confirmation is not None and password.get() != confirmation.get():
                feedback.configure(text="The passphrases do not match.")
                return
            try:
                vault = Vault(self.directory / "vault.sqlite3", password.get(), create=not exists)
            except (ValueError, VaultError, OSError, sqlite3.Error) as exc:
                feedback.configure(text=str(exc))
                return
            password.delete(0, "end")
            if confirmation is not None:
                confirmation.delete(0, "end")
            self.vault = vault
            try:
                self._workspace()
            except (ValueError, VaultError, OSError) as exc:
                vault.close()
                self.vault = None
                self._unlock_screen()
                messagebox.showerror("Cannot read vault", str(exc), parent=self.root)

        ttk.Button(
            screen,
            text="Unlock vault" if exists else "Create vault",
            style="Primary.TButton",
            command=unlock,
        ).pack(fill="x", pady=8)
        self.root.bind("<Return>", lambda _: unlock())
        ttk.Label(
            screen,
            text="There is no passphrase recovery. Keep a secure backup.\n"
            "Face verification is a demo and can be fooled by photos or video.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(20, 0))
        password.focus_set()

    def _workspace(self):
        self.root.unbind("<Return>")
        self._clear()
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, background=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        shell = ttk.Frame(canvas, padding=18)
        embedded = canvas.create_window((0, 0), window=shell, anchor="nw")
        shell.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(embedded, width=event.width))
        self.root.bind(
            "<MouseWheel>", lambda event: canvas.yview_scroll(-int(event.delta / 120), "units")
        )
        self.root.bind("<Button-4>", lambda _: canvas.yview_scroll(-1, "units"))
        self.root.bind("<Button-5>", lambda _: canvas.yview_scroll(1, "units"))
        header = ttk.Frame(shell)
        header.pack(fill="x", pady=(0, 16))
        ttk.Label(header, text="FACE DETECTION", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="LOCAL BIOMETRIC WORKSPACE", style="Muted.TLabel").pack(
            side="left", padx=20
        )
        ttk.Button(header, text="Lock vault", command=self.lock).pack(side="right")
        self.status = tk.StringVar(value="Ready. Start your camera to begin.")
        ttk.Label(
            shell, textvariable=self.status, wraplength=1050, font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 10))
        content = ttk.Frame(shell)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=0)
        content.rowconfigure(0, weight=1)
        video = ttk.Frame(content, style="Card.TFrame", padding=12)
        video.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        self.preview = tk.Label(
            video,
            bg=PANEL,
            fg=MUTED,
            text="CAMERA OFF\n\nStart the camera when you are ready.",
            font=("Segoe UI", 14),
            width=60,
        )
        self.preview.pack(fill="both", expand=True)
        controls = ttk.Frame(video, style="Card.TFrame")
        controls.pack(fill="x", pady=(10, 0))
        ttk.Label(controls, text="Camera", style="Card.TLabel").pack(side="left")
        ttk.Spinbox(controls, from_=0, to=9, textvariable=self.camera_index, width=3).pack(
            side="left", padx=8
        )
        self.start_button = ttk.Button(controls, text="Start camera", command=self.start_camera)
        self.start_button.pack(side="left", padx=4)
        ttk.Button(controls, text="Stop", command=self.stop_camera).pack(side="left", padx=4)
        actions = ttk.Frame(content, width=310)
        actions.grid(row=0, column=1, sticky="ns")
        modes = ttk.Notebook(actions)
        modes.pack(fill="x")
        enrol_panel = ttk.Frame(modes, padding=12)
        verify_panel = ttk.Frame(modes, padding=12)
        modes.add(enrol_panel, text="01  Enrol")
        modes.add(verify_panel, text="02  Verify")
        ttk.Label(enrol_panel, text="Person ID (e.g. KUNENE01)").pack(anchor="w")
        self.person_id = ttk.Entry(enrol_panel, width=30)
        self.person_id.pack(fill="x", pady=(4, 10))
        ttk.Label(enrol_panel, text="Display name").pack(anchor="w")
        self.name = ttk.Entry(enrol_panel, width=30)
        self.name.pack(fill="x", pady=(4, 8))
        self.consent = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            enrol_panel,
            variable=self.consent,
            text="I consent to storing my face\ntemplate on this PC.",
        ).pack(anchor="w")
        self.enrol_button = ttk.Button(
            enrol_panel, text="Enrol face", style="Primary.TButton", command=self.begin_enrol
        )
        self.enrol_button.pack(fill="x", pady=(8, 0))
        ttk.Label(verify_panel, text="Select the identity to verify").pack(anchor="w")
        self.selected = ttk.Combobox(verify_panel, state="readonly", width=28)
        self.selected.pack(fill="x", pady=(6, 10))
        self.verify_button = ttk.Button(
            verify_panel, text="Verify selected profile", command=self.begin_verify
        )
        self.verify_button.pack(fill="x")
        ttk.Label(
            verify_panel,
            text="Keep looking towards the camera.\n"
            "Five spaced matching frames\nare required to pass.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(16, 0))
        self.progress = ttk.Progressbar(actions, maximum=ENROLMENT_SAMPLES)
        self.progress.pack(fill="x", pady=(20, 10))
        ttk.Button(actions, text="Cancel capture", command=self.cancel).pack(fill="x")
        ttk.Label(
            actions,
            text="No liveness detection.\nPhotos or video may pass.",
            foreground=WARNING,
            wraplength=290,
        ).pack(anchor="w", pady=(16, 0))
        book = ttk.Notebook(shell, height=150)
        book.pack(fill="x", pady=(18, 0))
        profiles_tab = ttk.Frame(book, padding=8)
        activity_tab = ttk.Frame(book, padding=8)
        book.add(profiles_tab, text="Enrolled profiles")
        book.add(activity_tab, text="Recent activity (UTC)")
        self.profiles_tree = ttk.Treeview(
            profiles_tab, columns=("id", "name", "samples", "created"), show="headings", height=4
        )
        for col, title, width in [
            ("id", "Person ID", 150),
            ("name", "Display name", 220),
            ("samples", "Samples", 80),
            ("created", "Enrolled (UTC)", 230),
        ]:
            self.profiles_tree.heading(col, text=title)
            self.profiles_tree.column(col, width=width)
        self.profiles_tree.pack(side="left", fill="both", expand=True)
        ttk.Button(profiles_tab, text="Delete selected profile", command=self.delete_profile).pack(
            side="right", padx=10
        )
        self.events_tree = ttk.Treeview(
            activity_tab, columns=("time", "action", "id", "score"), show="headings", height=4
        )
        for col, title, width in [
            ("time", "Time (UTC)", 210),
            ("action", "Event", 210),
            ("id", "Person ID", 150),
            ("score", "Cosine similarity", 120),
        ]:
            self.events_tree.heading(col, text=title)
            self.events_tree.column(col, width=width)
        self.events_tree.pack(side="left", fill="both", expand=True)
        ttk.Button(activity_tab, text="Export recent CSV", command=self.export_events).pack(
            side="right", padx=10
        )
        footer = ttk.Frame(shell)
        footer.pack(fill="x", pady=(12, 0))
        self.model_button = ttk.Button(
            footer, text="Download / check models", command=self.download
        )
        self.model_button.pack(side="right")
        ttk.Label(
            footer,
            text="Encrypted templates  |  No face images saved  |  "
            f"SFace threshold {THRESHOLD:.2f}",
            style="Muted.TLabel",
        ).pack(side="left")
        self.refresh()
        try:
            self.engine = FaceEngine(self.models)
        except (OSError, ValueError, cv2.error):
            self.status.set("Download the OpenCV models once, then start your camera.")

    def refresh(self):
        profiles = self.vault.profiles()
        self.profiles_tree.delete(*self.profiles_tree.get_children())
        for profile in profiles:
            self.profiles_tree.insert(
                "",
                "end",
                iid=profile.person_id,
                values=(profile.person_id, profile.name, profile.samples, profile.created_at),
            )
        previous = self.selected.get()
        ids = [p.person_id for p in profiles]
        self.selected["values"] = ids
        self.selected.set(previous if previous in ids else (ids[0] if ids else ""))
        self.events_tree.delete(*self.events_tree.get_children())
        for event in self.vault.recent_events():
            score = "—" if event["score"] is None else f"{event['score']:.3f}"
            self.events_tree.insert(
                "",
                "end",
                values=(event["time_utc"], event["action"], event["person_id"] or "—", score),
            )

    def download(self):
        if self.downloading or self.mode is not None:
            return
        self.downloading = True
        self.model_button.state(["disabled"])

        def worker():
            try:
                ensure_models(self.models, lambda text: self.messages.put(("status", text)))
                engine = FaceEngine(self.models)
                self.messages.put(("models_ready", engine))
            except Exception as exc:
                self.messages.put(("models_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_messages(self):
        if self.closed:
            return
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if self.vault is None:
                    if kind in ("models_ready", "models_error"):
                        self.downloading = False
                        if kind == "models_ready":
                            self.engine = value
                    continue
                if kind == "status":
                    self.status.set(value)
                elif kind == "models_ready":
                    self.engine = value
                    self.downloading = False
                    self.model_button.state(["!disabled"])
                    self.status.set("Models verified. Start your camera to begin.")
                else:
                    self.downloading = False
                    self.model_button.state(["!disabled"])
                    self.status.set(f"Model download failed: {value}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def start_camera(self):
        if self.camera is not None:
            return
        if self.engine is None or self.downloading:
            self.status.set("Download and check the models first.")
            return
        try:
            camera = Camera(self.camera_index.get())
            camera.open()
        except (CameraError, ValueError, tk.TclError) as exc:
            self.status.set(str(exc))
            return
        self.camera = camera
        self.start_button.state(["disabled"])
        self.analysis = None
        self.status.set("Camera running. Enrol a profile or select one to verify.")
        self._tick()

    def stop_camera(self):
        if self.tick_job is not None:
            self.root.after_cancel(self.tick_job)
            self.tick_job = None
        self.cancel(quiet=True)
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        if self.vault is not None:
            self.start_button.state(["!disabled"])
            self.preview.configure(image="", text="CAMERA OFF\n\nNo capture is running.")
            self.preview.image = None
            self.status.set("Camera stopped.")

    def _ready(self) -> bool:
        if self.camera is None or self.engine is None:
            self.status.set("Start the camera first.")
            return False
        if self.mode is not None or self.downloading:
            self.status.set("Finish or cancel the current capture first.")
            return False
        return True

    def _busy(self, value: bool):
        for button in (self.enrol_button, self.verify_button):
            button.state(["disabled"] if value else ["!disabled"])

    def begin_enrol(self):
        if not self._ready():
            return
        try:
            validate_id(self.person_id.get())
            if not self.name.get().strip() or len(self.name.get().strip()) > 80:
                raise ValueError("Enter a display name of 1-80 characters.")
            if not self.consent.get():
                raise ValueError("Tick the enrolment consent box before capturing your face.")
            if self.person_id.get().strip() in [p.person_id for p in self.vault.profiles()]:
                raise ValueError("That Person ID exists. Delete it before re-enrolling.")
        except (ValueError, VaultError) as exc:
            self.status.set(str(exc))
            return
        self.enrol_id = self.person_id.get().strip()
        self.enrol_name = self.name.get().strip()
        self.mode = "enrol"
        self.samples = []
        self.last_sample = 0.0
        self.started = time.monotonic()
        self.progress.configure(maximum=ENROLMENT_SAMPLES, value=0)
        self._busy(True)

    def begin_verify(self):
        if not self._ready():
            return
        try:
            self.target = self.vault.get_profile(self.selected.get())
        except (ValueError, VaultError) as exc:
            self.status.set(str(exc))
            return
        self.started = time.monotonic()
        self.window = MatchWindow(started=self.started)
        self.last_score = None
        self.mode = "verify"
        self.progress.configure(maximum=self.window.required, value=0)
        self._busy(True)

    def cancel(self, quiet=False):
        was_active = self.mode is not None
        self.mode = None
        self.samples = []
        self.target = None
        self.window = None
        if self.vault is not None and hasattr(self, "progress"):
            self.progress.configure(value=0)
            self._busy(False)
            if was_active and not quiet:
                self.status.set("Capture cancelled. Unsaved samples were discarded.")

    def _process(self, frame, analysis, now):
        if self.mode == "enrol":
            if not self.consent.get():
                self.cancel(quiet=True)
                self.status.set("Consent withdrawn. No new profile was saved.")
                return
            if now - self.started >= CAPTURE_TIMEOUT:
                self.cancel(quiet=True)
                self.status.set("Enrolment timed out. No new profile was saved.")
                return
            if analysis.accepted and now - self.last_sample >= SAMPLE_INTERVAL:
                self.samples.append(self.engine.embedding(frame, analysis))
                self.last_sample = now
                self.progress.configure(value=len(self.samples))
            self.status.set(
                f"Enrolling {self.enrol_id}: {len(self.samples)}/"
                f"{ENROLMENT_SAMPLES} samples. {analysis.message}"
            )
            if len(self.samples) >= ENROLMENT_SAMPLES:
                template = make_template(self.samples)
                self.vault.add_profile(
                    self.enrol_id, self.enrol_name, template, len(self.samples), consent=True
                )
                self.cancel(quiet=True)
                self.consent.set(False)
                self.refresh()
                self.status.set("Enrolment complete. Select your profile and press Verify.")
        elif self.mode == "verify":
            score = None
            if analysis.accepted:
                score = cosine(self.engine.embedding(frame, analysis), self.target.embedding)
            self.last_score = score
            result = self.window.observe(score, now)
            self.progress.configure(value=self.window.matches)
            similarity = "—" if score is None else f"{score:.3f}"
            self.status.set(
                f"Verifying {self.target.person_id} | similarity {similarity} | "
                f"{self.window.matches}/{self.window.required} matches. "
                f"{analysis.message}"
            )
            if result != "waiting":
                target_id = self.target.person_id
                self.vault.record_verification(
                    target_id,
                    passed=result == "passed",
                    score=score,
                    threshold=self.window.threshold,
                )
                self.cancel(quiet=True)
                self.refresh()
                outcome = "VERIFIED" if result == "passed" else "NOT VERIFIED"
                self.status.set(f"{outcome}: {target_id}. Cosine similarity {similarity}.")

    def _tick(self):
        self.tick_job = None
        if self.closed or self.camera is None:
            return
        try:
            frame = self.camera.read()
            height, width = frame.shape[:2]
            if width > 960:
                frame = cv2.resize(frame, (960, int(height * 960 / width)))
            now = time.monotonic()
            if now - self.last_inference >= 0.15:
                self.analysis = self.engine.analyse(frame)
                self.last_inference = now
                if self.mode is not None:
                    self._process(frame, self.analysis, now)
            shown = frame.copy()
            if self.analysis is not None:
                colour = (202, 223, 102) if self.analysis.accepted else (128, 208, 255)
                for face in self.analysis.faces:
                    x, y, w, h = map(int, face[:4])
                    cv2.rectangle(shown, (x, y), (x + w, y + h), colour, 2)
            shown = cv2.flip(shown, 1)
            image = Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
            available = max(320, min(690, self.preview.winfo_width() - 4))
            photo = ImageTk.PhotoImage(ImageOps.contain(image, (available, 420)))
            self.preview.configure(image=photo, text="")
            self.preview.image = photo
        except CameraError as exc:
            self.stop_camera()
            self.status.set(str(exc))
            return
        except (ValueError, VaultError, OSError, sqlite3.Error, cv2.error) as exc:
            self.cancel(quiet=True)
            self.status.set(f"Capture stopped: {exc}")
        self.tick_job = self.root.after(30, self._tick)

    def delete_profile(self):
        if self.mode is not None:
            self.status.set("Cancel the active capture before deleting a profile.")
            return
        selected = self.profiles_tree.selection()
        if not selected:
            self.status.set("Select a row in Enrolled profiles first.")
            return
        person_id = selected[0]
        if messagebox.askyesno(
            "Delete profile", f"Delete {person_id} and its activity records?", parent=self.root
        ):
            try:
                self.vault.delete_profile(person_id)
                self.refresh()
                self.status.set("Profile and its activity records deleted.")
            except (ValueError, VaultError, OSError) as exc:
                self.status.set(str(exc))

    def export_events(self):
        if not messagebox.askyesno(
            "Export activity",
            "The CSV contains Person IDs and verification results in "
            "plain text. Save it in a private location?",
            parent=self.root,
        ):
            return
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".csv",
            initialfile="bioguard_activity_report.csv",
            filetypes=[("CSV file", "*.csv")],
        )
        if filename:
            try:
                rows = self.vault.recent_events(10000)
                with open(filename, "w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=[
                            "event_id",
                            "time_utc",
                            "action",
                            "person_id",
                            "score",
                            "threshold",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                self.status.set(f"Exported {len(rows)} recent events.")
            except (OSError, VaultError) as exc:
                self.status.set(f"Export failed: {exc}")

    def lock(self):
        self.stop_camera()
        if self.vault is not None:
            self.vault.close()
        self.vault = None
        self._unlock_screen()

    def close(self):
        self.closed = True
        if self.camera is not None:
            self.camera.close()
        if self.vault is not None:
            self.vault.close()
        self.root.destroy()


def launch(directory: Path, models: Path, camera_index: int = 0):
    cv2.setNumThreads(2)
    root = tk.Tk()
    Application(root, directory, models, camera_index)
    root.mainloop()
