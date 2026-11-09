# Security scope

Face-detection is a local educational face-verification demo. It has no presentation-attack
detection, challenge-response liveness, Windows authentication integration or secure
hardware binding. A photograph or replayed video may be accepted. Do not use it as the
sole control for a door, computer account, payment, employment decision or other important
access decision.

The vault protects stored profile payloads and activity details using authenticated
encryption. Its Person IDs and structure are visible. A weak passphrase permits offline
guessing. An attacker controlling the PC or the unlocked application can access decrypted
templates, modify software and alter/delete audit records. Python does not guarantee
secure erasure of keys from memory.

The UI is the management boundary: anyone with the vault passphrase can add and delete
profiles. There is no separate administrator account or remote service. Dependencies
and model downloads require Internet access during setup; recognition itself is local.

Profile deletion clears active related records, not backups. CSV exports are intentionally
plain text. Keep the OS account, backups and export destinations private. Do not attach
biometric databases, photographs, keys or passphrases to public bug reports.

If reporting a defect publicly, use synthetic embeddings and a temporary test vault.
For an actual deployment, independently review consent, retention, local requirements,
population testing, calibrated thresholds, spoof resistance and an alternate login path.
