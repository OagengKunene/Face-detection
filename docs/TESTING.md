# Testing Face-detection

## Automated checks without a webcam

Install `requirements-dev.txt`, then run:

```text
python -m pytest -q -m "not integration"
python run.py self-test
python -m ruff check .
python -m ruff format --check .
```

The suite covers matching streaks and timeout, invalid vectors and IDs, mixed enrolment
samples, consent and its withdrawal, encrypted persistence, wrong passphrases, duplicate
IDs, modified ciphertext, swapped profile records, deletion, bad model downloads,
camera disconnection / release, and real desktop-controller calls with synthetic embeddings.

The native Tk widget smoke test runs only when a desktop display is available (including
Windows CI). A skipped native GUI test is not a successful GUI test.

## Real OpenCV inference

The optional integration tests use two public sample portraits. They never contact a
biometric service or write the portraits into the vault. Download them explicitly:

```text
python run.py models --model-dir models
python scripts/fetch_test_fixtures.py --destination test-fixtures
```

PowerShell:

```powershell
$env:BIOGUARD_MODELS = "models"
$env:BIOGUARD_FACE_A = "test-fixtures/portrait_a.png"
$env:BIOGUARD_FACE_B = "test-fixtures/portrait_b.jpg"
python -m pytest -q -m integration -s
```

Linux:

```bash
BIOGUARD_MODELS=models \
BIOGUARD_FACE_A=test-fixtures/portrait_a.png \
BIOGUARD_FACE_B=test-fixtures/portrait_b.jpg \
python -m pytest -q -m integration -s
```

These tests run the real detector and recognizer, aggregate an encrypted template,
verify a mildly modified image of the same portrait, reject the other portrait,
exercise the five-frame decision, and delete the temporary profile. A blank frame and
a composite with two faces must also be rejected. Portrait A is resized 1.5× so that its
face exceeds the production minimum of 100 px; the quality gate is not weakened.

This is a software-path check, not a study of biometric performance. Eight augmented
samples from one photo do not replace independent live captures, and two subjects cannot
measure population false-match rates or fairness. No liveness or spoof resistance is claimed.

## Test the actual Windows PC

1. Fully extract the ZIP, install 64-bit Python 3.12–3.14 with Tcl/Tk, and close other
   applications that may hold the webcam.
2. Run `Test_PC.bat` (or `Test_PC.bat 1` for a second camera).
3. Confirm all checks report `PASS`. Keep `pc_test_report.json` if something fails.
4. Open `Start_Face_Detection.bat`, create a vault, and make sure camera preview and all controls
   are visible. The workspace scrolls vertically on smaller screens.
5. Check consent, enrol yourself under a new Person ID, then verify that selected profile.
6. Verify the same profile with a different consenting person. The result should fail.
7. Cover the camera, move too far away, introduce a second person, and try poor lighting.
   Capture should show guidance and refuse unsuitable samples. Remove the second person
   before resuming verification.
8. Start an enrolment, then cancel it or untick consent. Confirm no partial profile appears.
9. Stop and restart the camera. Lock the vault and check that the feed stops.
10. Close/reopen the app. Confirm the wrong vault passphrase is rejected and the correct
    passphrase restores the profile.
11. Check that recent activity shows completed verification outcomes. Export only if a
    plain-text CSV is appropriate for your private test folder.
12. Delete your test profile. Confirm its related activity is removed from the active vault.

If a camera test fails, check Windows **Settings → Privacy & security → Camera**, allow
desktop apps, close Teams/Zoom, reconnect the device, then try index 1. If Python reports
`No module named tkinter`, modify/reinstall CPython with Tcl/Tk included.

The app has no passphrase reset. Use a separate test data directory if you want to start
over without changing an existing vault:

```text
python run.py gui --data-dir test-vault
```

Only enrol people who agree to the local test. Do not upload face images, templates or
vault files when asking for help. The PC diagnostic report contains no such data.
