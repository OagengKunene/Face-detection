# Validation record · 9 November 2026

This records the tests actually run before delivery, not future checks.

| Check | Observed result |
| --- | --- |
| Runtime | Cloud Linux x86_64, Python 3.12.14, OpenCV 5.0.0 |
| Complete pytest suite with real-model fixtures enabled | **86 passed, 1 skipped**, 6.07 s |
| Native Tk widget smoke test | **Skipped**: no usable desktop display in this runtime |
| Real model downloads | Both sizes and SHA-256 hashes match the pinned OpenCV Zoo objects |
| Enrol / verify / nonmatch / delete integration | Passed using real YuNet and SFace inference and a temporary encrypted vault |
| Same-portrait similarity after mild image changes | 0.9986263514 |
| Different-portrait similarity | 0.1453854740 |
| Decision threshold | 0.45; same portrait passed and different portrait failed |
| Blank image and a two-face composite | Rejected by the actual detector / quality gate |
| Standalone offline self-test | Six checks passed |
| Ruff lint and format checks | Passed |
| Wheel and source distribution build | Passed; wheel installed into a separate environment and its version / six self-tests passed |
| Windows Python 3.14 dependency availability | Matching Windows x64 wheels confirmed for all four pinned runtime dependencies |
| Live camera on the cloud machine | Unavailable; camera 0 diagnostic returned FAIL and no frame was captured |
| User's Windows PC / webcam | **Not tested remotely**; no connection to that PC was available |
| GitHub Actions Windows / Linux matrix | Configured, not yet run at the time of this record |

The complete suite includes numerical validation, encrypted storage, consent, changed
consent during capture, temporal matching, real controller methods, model corruption,
camera lifecycle and two real-model integration tests. Mock camera tests do not establish
that a physical camera works. Controller tests do not establish a rendered native UI works.

The two portrait scores are checks of this specific software path. They are not measured
population accuracy, false-acceptance rates or live webcam results. Enrolment samples in
the integration test are augmentations of one public sample photograph. The application
has no liveness detection and a photograph can pass.

For the remaining local tests, run `Test_PC.bat`, then follow [TESTING.md](TESTING.md).
Keep personal face images and vaults local; the diagnostic JSON does not contain them.
