# Face-detection · Python + OpenCV

A desktop biometric project for **face enrolment and one-to-one identity verification**.
Choose an enrolled Person ID, look at the webcam, and receive a verified / not verified
result. Recognition runs locally on the CPU using OpenCV YuNet and SFace.

**This is a portfolio and learning application. It has no liveness detection and can be
fooled by a photograph or video. It does not unlock Windows or control physical access.**

## What it does

- Native desktop interface with a mirrored camera preview and face guidance.
- Consent-based enrolment: eight quality-checked samples become one 128-value template.
- Verification against one selected profile: five spaced matching frames are required.
- Rejection of missing faces, multiple faces, small / clipped faces, blur and poor lighting.
- Passphrase-encrypted face templates, display names and activity payloads in local SQLite.
- Profile deletion, recent activity, and an explicit plain-text CSV export.
- Windows launchers, an offline self-test, camera diagnostics and automated tests.
- Pinned model downloads with exact size and SHA-256 checks. No biometric cloud API.

Repository: [OagengKunene/Face-detection](https://github.com/OagengKunene/Face-detection)

## Start on Windows

Use **64-bit CPython 3.12, 3.13 or 3.14** from [python.org](https://www.python.org/downloads/windows/).
Include **pip** and **Tcl/Tk** in the installer. Python 3.14 is supported by the pinned
dependency wheels. No compiler, CUDA, dlib, API key or paid service is needed.

1. Clone the repository or download and fully extract its ZIP.
2. Double-click **`Start_Face_Detection.bat`**.
3. On the first run, dependencies and approximately 39 MB of OpenCV models are downloaded.
4. Create a vault passphrase of at least 12 characters. There is no password recovery.
5. Click **Start camera**. Use **01 Enrol**, enter `KUNENE01` and your display name,
   tick the consent box, then click **Enrol face**. Keep only your face in the frame.
6. Open **02 Verify**, select your Person ID and click **Verify selected profile**.

Subsequent recognition is offline once dependencies and models are present. A new camera
does not need model training: enrolment stores the local SFace template.

### Test your PC

Double-click **`Test_PC.bat`**. It installs/checks dependencies and models, runs six offline
self-tests, opens the webcam, checks a real captured frame and releases the device.
It writes **`pc_test_report.json`** beside the launcher. This diagnostic file contains
software versions, check results and camera dimensions, **no face images or templates**.

If you use an external camera, open Command Prompt in this folder and run:

```bat
Test_PC.bat 1
```

The PC diagnostic proves device access, not biometric accuracy. Complete the manual
enrolment / verification checklist in [docs/TESTING.md](docs/TESTING.md) as well.

## Run from a terminal

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py models
.venv\Scripts\python.exe run.py gui
```

On Linux, use `python3 -m venv .venv` and `.venv/bin/python` instead. Tkinter and a desktop
display must be available; on Debian-family systems Tkinter is usually `python3-tk`.
OpenCV is the **headless wheel** because Tkinter, rather than OpenCV HighGUI, draws the
interface. VideoCapture and the DNN recognition APIs are included in this wheel.

Other commands:

```text
python run.py --version
python run.py self-test
python run.py doctor --camera 0 --output pc_test_report.json
python run.py gui --camera 1
python run.py gui --data-dir PRIVATE_FOLDER --model-dir MODELS_FOLDER
python -m bioguard self-test
```

Every command returns a nonzero exit code on failure. `doctor` skips the camera unless
`--camera` is supplied. It checks the Tkinter import; it does not claim that the GUI or
a real user's face has been tested.

## Recognition and storage

| Stage | Implementation |
| --- | --- |
| Capture | Explicitly started local webcam; frames stay in memory |
| Detection | OpenCV YuNet, one face required, detection score at least 0.9 |
| Quality | Minimum 100 px face; no clipped box; grayscale mean 45–210; Laplacian variance ≥45 |
| Representation | SFace aligned crop → normalised 128-value embedding |
| Enrolment | Eight spaced samples; consistency checks; normalised mean template |
| Decision | Cosine similarity ≥0.45 across five frames at least 0.20 s apart |
| Timeout | Verification fails at 15 s; unfinished enrolment is discarded at 25 s |
| Persistence | SQLite with Fernet encryption; PBKDF2-HMAC-SHA256, 600,000 iterations and random salt |

The `0.45` threshold is a project design choice, stricter than the `0.363` LFW example
in the [OpenCV tutorial](https://docs.opencv.org/4.11.0/d0/dd4/tutorial_dnn_face.html).
Cosine similarity is **not a probability or confidence percentage**. Calibrate it on
consenting, representative local subjects before relying on results. The published
OpenCV benchmark is not this project's measured accuracy.

The vault is created outside the source tree. The storage folder keeps its original name
so an existing vault from the earlier download remains available:

- Windows: `%LOCALAPPDATA%\BioGuard\vault.sqlite3`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/bioguard/vault.sqlite3`
- Models: the `models` subfolder beside the vault, or `--model-dir`.

Person IDs, row counts and the database schema are visible in SQLite; display names,
embeddings, consent timestamps and activity details are encrypted. A passphrase-derived
key exists in memory while unlocked and is not written to a key file. Raw photographs
and video are never saved by the application. Face embeddings remain sensitive data.

Use a strong, unique vault passphrase. Back up the closed vault privately, and keep its
passphrase separately. Deletion removes the profile and its related active event rows;
it cannot erase external backups or guarantee physical erasure on an SSD. The local
audit is editable by someone who controls the PC; it is not a tamper-proof audit service.
Anyone with the vault passphrase can manage profiles. The app is not an OS login provider.

## Project layout

| Path | Purpose |
| --- | --- |
| `bioguard/gui.py` | Native interface, enrolment and verification controller |
| `bioguard/vision.py` | Detection, quality gates, alignment and embedding |
| `bioguard/decisions.py` | Numeric checks, template construction and temporal matching |
| `bioguard/storage.py` | Encrypted vault, profile lifecycle and activity records |
| `bioguard/models.py` | Immutable, checksum-verified model downloads |
| `bioguard/camera.py` | Camera acquisition, Windows fallback and release |
| `bioguard/diagnostics.py` | PC diagnostics and offline self-test |
| `tests/` | Storage, decisions, camera, controller and real-model tests |
| `.github/workflows/ci.yml` | Windows / Linux automated checks and a real-model test job |

## Development

```text
python -m pip install -r requirements-dev.txt
python -m pytest -m "not integration" -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

See [docs/TESTING.md](docs/TESTING.md) for real-model tests and the local PC checklist,
[docs/TEST_RESULTS.md](docs/TEST_RESULTS.md) for the actual validation record, and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for model and test-image provenance.

The source is MIT-licensed. OpenCV and its downloaded models keep their upstream licenses.
Do not commit vaults, private images, credentials or exported activity. The `.gitignore`
excludes runtime data, models and diagnostic reports.
