"""Export local YOLO26 detection weights for the ONNX analysis service."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[1280], metavar="SIZE",
                        help="square size, or HEIGHT WIDTH for a fixed rectangular input such as 736 1280")
    parser.add_argument("--head", choices=("accuracy", "end2end"), default="accuracy")
    parser.add_argument("--half", action="store_true", help="FP16 export; requires CUDA")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"local checkpoint not found: {args.checkpoint}")
    if args.output.suffix.lower() != ".onnx" or args.output.exists():
        parser.error("output must be a new .onnx file")
    if len(args.imgsz) > 2 or any(size <= 0 or size % 32 for size in args.imgsz):
        parser.error("imgsz must be one or two positive multiples of 32")
    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz

    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    if args.half and not torch.cuda.is_available():
        parser.error("FP16 export requires CUDA; no FP16 artifact was produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="yolo26-export-", dir=args.output.parent) as temporary:
        checkpoint = Path(temporary) / args.checkpoint.name
        shutil.copyfile(args.checkpoint, checkpoint)
        model = YOLO(str(checkpoint))
        if model.task != "detect" or not hasattr(model.model, "end2end"):
            parser.error("a YOLO26 detection checkpoint with selectable heads is required")
        options = dict(format="onnx", imgsz=imgsz, dynamic=False, simplify=False, opset=17,
                       device=0 if args.half else "cpu")
        # Ultralytics 8.4.60 uses end2end; newer releases use nms to select the head.
        if "end2end" in DEFAULT_CFG_DICT:
            options.update(end2end=args.head == "end2end", nms=False)
        else:
            options["nms"] = False if args.head == "end2end" else None
        if args.half:
            options["quantize" if "quantize" in DEFAULT_CFG_DICT else "half"] = 16 if "quantize" in DEFAULT_CFG_DICT else True
        exported = Path(model.export(**options))
        import onnx

        artifact = onnx.load(str(exported))
        metadata = {entry.key: entry.value for entry in artifact.metadata_props}
        expected = str(args.head == "end2end")
        if metadata.get("end2end") != expected:
            raise RuntimeError(f"requested {args.head} head, exported metadata: {metadata}")
        exported.replace(args.output)
    print(f"Exported {args.head} {'x'.join(map(str, args.imgsz))}px ONNX: {args.output}")


if __name__ == "__main__":
    main()
