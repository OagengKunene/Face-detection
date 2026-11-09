"""Passphrase-encrypted templates and audit payloads in local SQLite."""

import base64
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

from bioguard.config import MODEL_ID
from bioguard.decisions import unit_vector, validate_id

ITERATIONS = 600_000
VERIFIER = "BioGuard vault version 1"


class VaultError(Exception):
    """The vault cannot be safely opened or read."""


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Profile:
    person_id: str
    name: str
    embedding: np.ndarray
    samples: int
    created_at: str


class Vault:
    def __init__(self, path: Path, passphrase: str, *, create: bool = False):
        self.path = Path(path)
        self._db: sqlite3.Connection | None = None
        self._fernet: Fernet | None = None
        if create:
            if len(passphrase) < 12:
                raise ValueError("Choose a vault passphrase of at least 12 characters.")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        if not self.path.exists():
            raise VaultError("Vault does not exist. Create one in the application.")
        try:
            self._db = sqlite3.connect(self.path, timeout=5)
            self._db.execute("PRAGMA secure_delete = ON")
            self._db.execute("PRAGMA journal_mode = DELETE")
            self._db.execute("PRAGMA foreign_keys = ON")
            if create:
                salt = os.urandom(16)
                self._fernet = self._derive(passphrase, salt)
                with self._db:
                    self._db.executescript(
                        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value BLOB NOT NULL);"
                        "CREATE TABLE profiles (person_id TEXT PRIMARY KEY, payload BLOB NOT NULL);"
                        "CREATE TABLE events (id INTEGER PRIMARY KEY, person_id TEXT, "
                        "payload BLOB NOT NULL);"
                        "PRAGMA user_version = 1;"
                    )
                    self._db.executemany(
                        "INSERT INTO metadata VALUES (?, ?)",
                        [("salt", salt), ("verifier", self._fernet.encrypt(VERIFIER.encode()))],
                    )
            else:
                version = self._db.execute("PRAGMA user_version").fetchone()[0]
                if version != 1:
                    raise VaultError("Unsupported or damaged vault version.")
                values = dict(self._db.execute("SELECT key, value FROM metadata"))
                if len(values["salt"]) != 16:
                    raise VaultError("Damaged vault metadata.")
                self._fernet = self._derive(passphrase, values["salt"])
                if self._fernet.decrypt(values["verifier"]).decode() != VERIFIER:
                    raise VaultError("Wrong passphrase or damaged vault.")
        except (sqlite3.Error, InvalidToken, KeyError, UnicodeError, VaultError) as exc:
            self.close()
            raise VaultError("Cannot unlock vault: wrong passphrase or damaged database.") from exc

    @staticmethod
    def _derive(passphrase: str, salt: bytes) -> Fernet:
        key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, ITERATIONS, dklen=32)
        return Fernet(base64.urlsafe_b64encode(key))

    def _connection(self) -> sqlite3.Connection:
        if self._db is None or self._fernet is None:
            raise VaultError("Vault is locked.")
        return self._db

    def _encrypt(self, payload: dict) -> bytes:
        self._connection()
        return self._fernet.encrypt(json.dumps(payload, allow_nan=False).encode())

    def _decrypt(self, payload: bytes) -> dict:
        self._connection()
        try:
            data = json.loads(self._fernet.decrypt(payload))
            if not isinstance(data, dict):
                raise ValueError("Invalid payload")
            return data
        except (InvalidToken, ValueError, UnicodeError) as exc:
            raise VaultError("A vault record is damaged. Restore a trusted backup.") from exc

    def add_profile(
        self, person_id: str, name: str, embedding: np.ndarray, samples: int, *, consent: bool
    ) -> None:
        if not consent:
            raise ValueError("Explicit enrolment consent is required.")
        person_id = validate_id(person_id)
        name = name.strip()
        if not name or len(name) > 80 or any(ord(char) < 32 for char in name):
            raise ValueError("Display name must contain 1-80 printable characters.")
        if not 8 <= samples <= 30:
            raise ValueError("Enrolment sample count must be 8-30.")
        payload = self._encrypt(
            {
                "person_id": person_id,
                "name": name,
                "embedding": unit_vector(embedding).tolist(),
                "samples": samples,
                "created_at": timestamp(),
                "consented_at": timestamp(),
                "model_id": MODEL_ID,
            }
        )
        db = self._connection()
        try:
            with db:
                db.execute("INSERT INTO profiles VALUES (?, ?)", (person_id, payload))
                self._insert_event("enrolled", person_id)
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "That Person ID already exists. Delete it before re-enrolling."
            ) from exc

    def get_profile(self, person_id: str) -> Profile:
        person_id = validate_id(person_id)
        row = (
            self._connection()
            .execute("SELECT payload FROM profiles WHERE person_id = ?", (person_id,))
            .fetchone()
        )
        if row is None:
            raise ValueError("Select an enrolled profile first.")
        payload = self._decrypt(row[0])
        try:
            if payload["person_id"] != person_id or payload["model_id"] != MODEL_ID:
                raise VaultError("Profile identity or model version mismatch.")
            return Profile(
                person_id,
                payload["name"],
                unit_vector(payload["embedding"]),
                int(payload["samples"]),
                payload["created_at"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise VaultError("Invalid profile data.") from exc

    def profiles(self) -> list[Profile]:
        ids = self._connection().execute("SELECT person_id FROM profiles ORDER BY person_id")
        return [self.get_profile(row[0]) for row in ids.fetchall()]

    def _insert_event(
        self,
        action: str,
        person_id: str | None,
        score: float | None = None,
        threshold: float | None = None,
    ) -> None:
        payload = self._encrypt(
            {
                "time_utc": timestamp(),
                "action": action,
                "person_id": person_id,
                "score": score,
                "threshold": threshold,
            }
        )
        self._connection().execute(
            "INSERT INTO events (person_id, payload) VALUES (?, ?)", (person_id, payload)
        )

    def record_verification(
        self, person_id: str, *, passed: bool, score: float | None, threshold: float
    ) -> None:
        self.get_profile(person_id)
        if score is not None and (not np.isfinite(score) or not -1 <= score <= 1):
            raise ValueError("Invalid verification score.")
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Invalid verification threshold.")
        with self._connection():
            self._insert_event(
                "verification_passed" if passed else "verification_failed",
                person_id,
                score,
                threshold,
            )

    def recent_events(self, limit: int = 200) -> list[dict]:
        if not 1 <= limit <= 10000:
            raise ValueError("Event limit must be between 1 and 10000.")
        rows = self._connection().execute(
            "SELECT id, person_id, payload FROM events ORDER BY id DESC LIMIT ?", (limit,)
        )
        events = []
        for event_id, person_id, raw in rows:
            payload = self._decrypt(raw)
            if payload.get("person_id") != person_id:
                raise VaultError("Audit record identity mismatch.")
            events.append({"event_id": event_id, **payload})
        return events

    def delete_profile(self, person_id: str) -> None:
        self.get_profile(person_id)
        with self._connection() as db:
            db.execute("DELETE FROM events WHERE person_id = ?", (person_id,))
            db.execute("DELETE FROM profiles WHERE person_id = ?", (person_id,))
            self._insert_event("profile_deleted", None)

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
        self._db = None
        self._fernet = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
