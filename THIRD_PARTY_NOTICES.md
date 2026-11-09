# Third-party notices

Face-detection's original application code is MIT-licensed. Third-party packages and model
weights are not relicensed by that license. Dependencies are installed from PyPI;
weights are downloaded from OpenCV's official repository rather than committed here.

## OpenCV model provenance

Pinned OpenCV Zoo revision: `48b10fa505aeb51fbd127fc60a5c660040e823b7`.
`bioguard/models.py` records the immutable download locations, exact sizes and SHA-256
hashes (the upstream Git LFS object identifiers).

| Model | Upstream documentation and license |
| --- | --- |
| YuNet, `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet) and its LICENSE |
| SFace, `face_recognition_sface_2021dec.onnx` | [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface) and its Apache-2.0 LICENSE |

The API pattern follows the official [OpenCV face detection / recognition tutorial](https://docs.opencv.org/4.11.0/d0/dd4/tutorial_dnn_face.html).
No third-party application implementation is copied into this project.

## Optional test portraits

`scripts/fetch_test_fixtures.py` downloads these only when explicitly invoked. The files
are excluded from Git and the deliverable ZIP. They are software test inputs, not data
used to train these models or to enrol a real user in the delivered application.

| Local fixture | Upstream sample |
| --- | --- |
| `portrait_a.png` | scikit-image v0.24.0 `astronaut.png`, a NASA astronaut portrait; [sample documentation](https://scikit-image.org/docs/0.24.x/api/skimage.data.html#skimage.data.astronaut) identifies it as public domain |
| `portrait_b.jpg` | Matplotlib v3.10.7 `grace_hopper.jpg`, the US Navy portrait distributed with Matplotlib; [sample data source](https://github.com/matplotlib/matplotlib/tree/v3.10.7/lib/matplotlib/mpl-data/sample_data) |

The automated tests call them fixture A and fixture B. They do not label, search for, or
identify unknown people. Their consent / legal status does not authorize collecting
faces from other people for a real deployment.
