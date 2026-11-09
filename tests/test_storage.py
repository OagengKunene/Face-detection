import sqlite3

import numpy as np
import pytest

from bioguard.storage import Vault, VaultError


def enrol(vault, vector, person_id="TEST01", name="Private display name"):
    vault.add_profile(person_id, name, vector, 8, consent=True)


def test_profile_and_activity_roundtrip(vault, vector):
    enrol(vault, vector)
    profile = vault.get_profile("TEST01")
    assert profile.name == "Private display name"
    assert profile.samples == 8
    assert np.array_equal(profile.embedding, vector)
    vault.record_verification("TEST01", passed=True, score=0.85, threshold=0.45)
    events = vault.recent_events()
    assert [e["action"] for e in events] == ["verification_passed", "enrolled"]
    assert events[0]["score"] == 0.85


def test_database_does_not_contain_plaintext_sensitive_payloads(vault, vector):
    enrol(vault, vector)
    vault.record_verification("TEST01", passed=False, score=0.01, threshold=0.45)
    raw = vault.path.read_bytes()
    assert b"Private display name" not in raw
    assert b"embedding" not in raw
    assert b"verification_failed" not in raw
    assert b"a private testing passphrase" not in raw


def test_reopen_and_wrong_password(vault, vector):
    enrol(vault, vector)
    with pytest.raises(VaultError):
        Vault(vault.path, "incorrect secret")
    with Vault(vault.path, "a private testing passphrase") as reopened:
        assert reopened.get_profile("TEST01").name == "Private display name"


def test_consent_is_required(vault, vector):
    with pytest.raises(ValueError, match="consent"):
        vault.add_profile("TEST01", "Test", vector, 8, consent=False)
    assert not vault.profiles()


def test_short_passphrase_creates_no_file(tmp_path):
    path = tmp_path / "vault.sqlite3"
    with pytest.raises(ValueError, match="12"):
        Vault(path, "short", create=True)
    assert not path.exists()


def test_create_never_overwrites_existing_vault(vault):
    original = vault.path.read_bytes()
    with pytest.raises(FileExistsError):
        Vault(vault.path, "a new testing passphrase", create=True)
    assert vault.path.read_bytes() == original


def test_missing_vault_and_corrupt_database(tmp_path):
    path = tmp_path / "vault.sqlite3"
    with pytest.raises(VaultError):
        Vault(path, "testing passphrase")
    path.write_text("not a database")
    with pytest.raises(VaultError):
        Vault(path, "testing passphrase")


def test_duplicate_id_does_not_replace_profile(vault, vector):
    enrol(vault, vector)
    with pytest.raises(ValueError, match="already exists"):
        enrol(vault, vector, name="Replacement")
    assert vault.get_profile("TEST01").name == "Private display name"
    assert len(vault.recent_events()) == 1


def test_delete_removes_related_events_but_keeps_other_people(vault, vector):
    enrol(vault, vector, "TEST01")
    enrol(vault, vector, "TEST02")
    vault.record_verification("TEST01", passed=False, score=0.1, threshold=0.45)
    vault.delete_profile("TEST01")
    assert [p.person_id for p in vault.profiles()] == ["TEST02"]
    assert all(e["person_id"] != "TEST01" for e in vault.recent_events())
    assert vault.recent_events()[0]["action"] == "profile_deleted"


def test_tampered_ciphertext_is_rejected(vault, vector):
    enrol(vault, vector)
    with sqlite3.connect(vault.path) as db:
        db.execute("UPDATE profiles SET payload = ?", (b"tampered",))
    with pytest.raises(VaultError, match="damaged"):
        vault.get_profile("TEST01")


def test_swapped_profile_ciphertexts_are_rejected(vault, vector):
    enrol(vault, vector, "TEST01")
    enrol(vault, vector, "TEST02")
    with sqlite3.connect(vault.path) as db:
        raw = db.execute("SELECT payload FROM profiles WHERE person_id='TEST01'").fetchone()[0]
        db.execute("UPDATE profiles SET payload=? WHERE person_id='TEST02'", (raw,))
    with pytest.raises(VaultError, match="mismatch"):
        vault.get_profile("TEST02")


def test_record_for_unknown_profile_is_rejected(vault):
    with pytest.raises(ValueError):
        vault.record_verification("UNKNOWN01", passed=True, score=0.8, threshold=0.45)


def test_closed_vault_is_unusable(vault):
    vault.close()
    with pytest.raises(VaultError, match="locked"):
        vault.profiles()


@pytest.mark.parametrize("name", ["", "n" * 81, "control\ncharacter"])
def test_invalid_names(vault, vector, name):
    with pytest.raises(ValueError):
        enrol(vault, vector, name=name)
