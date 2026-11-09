"""Command-line entry points for the desktop app, setup and local PC tests."""

import argparse
import json
import sys
from pathlib import Path

from bioguard import __version__
from bioguard.config import data_dir, model_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Face-detection local biometric workspace")
    parser.add_argument("--version", action="version", version=f"Face-detection {__version__}")
    sub = parser.add_subparsers(dest="command")
    gui = sub.add_parser("gui", help="Open the desktop app (also the default)")
    gui.add_argument("--camera", type=int, default=0, choices=range(10))
    gui.add_argument("--data-dir", type=Path, default=data_dir())
    gui.add_argument("--model-dir", type=Path, default=model_dir())
    models = sub.add_parser("models", help="Download and verify the two OpenCV ONNX models")
    models.add_argument("--model-dir", type=Path, default=model_dir())
    diagnostic = sub.add_parser("doctor", help="Check Python, models, DNN inference and webcam")
    diagnostic.add_argument("--camera", type=int, choices=range(10))
    diagnostic.add_argument("--model-dir", type=Path, default=model_dir())
    diagnostic.add_argument("--output", type=Path, help="Save a diagnostic JSON report")
    sub.add_parser("self-test", help="Run offline encryption/storage/verification checks")
    args = parser.parse_args(argv)
    try:
        if args.command == "models":
            from bioguard.models import ensure_models

            ensure_models(args.model_dir)
        elif args.command == "doctor":
            from bioguard.diagnostics import doctor

            report = doctor(args.model_dir, args.camera)
            rendered = json.dumps(report, indent=2)
            print(rendered)
            if args.output:
                args.output.write_text(rendered + "\n", encoding="utf-8")
            return 0 if report["ok"] else 1
        elif args.command == "self-test":
            from bioguard.diagnostics import self_test

            results = self_test()
            for result in results:
                print(f"PASS  {result}")
            print(f"\n{len(results)} self-tests passed. This does not test a real face or webcam.")
        else:
            from bioguard.gui import launch

            launch(
                getattr(args, "data_dir", data_dir()),
                getattr(args, "model_dir", model_dir()),
                getattr(args, "camera", 0),
            )
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Face-detection: {exc}", file=sys.stderr)
        if args.command in (None, "gui"):
            print("Run 'python run.py doctor --camera 0' for diagnostics.", file=sys.stderr)
        return 1
    return 0
