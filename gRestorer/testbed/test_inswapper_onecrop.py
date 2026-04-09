import argparse
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


def parse_args():
    p = argparse.ArgumentParser(description="Standalone one-crop InsightFace InSwapper test")
    p.add_argument("--source", required=True, help="Path to source face image")
    p.add_argument("--target", required=True, help="Path to target crop image")
    p.add_argument("--swap-model", required=True, help="Path to inswapper ONNX model")
    p.add_argument("--provider", choices=["cuda", "cpu"], default="cuda", help="Execution provider")
    p.add_argument("--det-size", type=int, default=640, help="FaceAnalysis detection size")
    p.add_argument("--outdir", default=".", help="Directory for outputs")
    return p.parse_args()


def make_app_and_swapper(provider: str, swap_model: str, det_size: int):
    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    swapper = get_model(swap_model, providers=providers)
    return app, swapper, providers, ctx_id


def pick_largest_face(faces):
    return max(
        faces,
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
    )


def draw_face(img: np.ndarray, face, color=(0, 255, 0)) -> np.ndarray:
    vis = img.copy()
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    if getattr(face, "kps", None) is not None:
        for p in face.kps:
            px, py = int(p[0]), int(p[1])
            cv2.circle(vis, (px, py), 2, (0, 0, 255), -1)
    return vis


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    app, swapper, providers, ctx_id = make_app_and_swapper(args.provider, args.swap_model, args.det_size)

    src = cv2.imread(args.source, cv2.IMREAD_COLOR)
    tgt = cv2.imread(args.target, cv2.IMREAD_COLOR)
    assert src is not None, args.source
    assert tgt is not None, args.target

    src_faces = app.get(src)
    tgt_faces = app.get(tgt)

    print("providers:", providers)
    print("ctx_id:", ctx_id)
    print("src faces:", len(src_faces))
    print("tgt faces:", len(tgt_faces))

    assert src_faces, f"No source face detected: {args.source}"
    assert tgt_faces, f"No target face detected: {args.target}"

    src_face = pick_largest_face(src_faces)
    tgt_face = pick_largest_face(tgt_faces)

    print("src bbox:", src_face.bbox)
    print("tgt bbox:", tgt_face.bbox)
    print("src emb norm:", np.linalg.norm(src_face.normed_embedding))
    print("tgt emb norm:", np.linalg.norm(tgt_face.normed_embedding))

    cv2.imwrite(str(outdir / "source_detected.png"), draw_face(src, src_face))
    cv2.imwrite(str(outdir / "target_detected.png"), draw_face(tgt, tgt_face))

    out_pb = swapper.get(tgt.copy(), tgt_face, src_face, paste_back=True)
    out_np = swapper.get(tgt.copy(), tgt_face, src_face, paste_back=False)

    print("type(out_pb):", type(out_pb))
    if hasattr(out_pb, "shape"):
        print("out_pb.shape:", out_pb.shape, "dtype:", out_pb.dtype)

    print("type(out_np):", type(out_np))

    aligned = None
    affine = None
    if isinstance(out_np, tuple):
        print("len(out_np):", len(out_np))
        for i, x in enumerate(out_np):
            print(f"  out_np[{i}] type:", type(x))
            if hasattr(x, "shape"):
                print(f"  out_np[{i}] shape:", x.shape, "dtype:", x.dtype)
        if len(out_np) >= 1 and isinstance(out_np[0], np.ndarray):
            aligned = out_np[0]
        if len(out_np) >= 2 and isinstance(out_np[1], np.ndarray):
            affine = out_np[1]
    elif isinstance(out_np, list):
        print("len(out_np):", len(out_np))
        for i, x in enumerate(out_np):
            print(f"  out_np[{i}] type:", type(x))
            if hasattr(x, "shape"):
                print(f"  out_np[{i}] shape:", x.shape, "dtype:", x.dtype)
        if len(out_np) >= 1 and isinstance(out_np[0], np.ndarray):
            aligned = out_np[0]
    elif hasattr(out_np, "shape"):
        print("out_np.shape:", out_np.shape, "dtype:", out_np.dtype)
        aligned = out_np
    else:
        print("out_np repr:", repr(out_np))

    if isinstance(out_pb, np.ndarray):
        cv2.imwrite(str(outdir / "standalone_pasteback_true.png"), out_pb)
        diff = cv2.absdiff(tgt, out_pb)
        cv2.imwrite(str(outdir / "standalone_diff.png"), diff)
        print("paste_back_true mean abs diff:", float(np.mean(np.abs(out_pb.astype(np.int16) - tgt.astype(np.int16)))))

    if aligned is not None:
        cv2.imwrite(str(outdir / "standalone_pasteback_false_aligned.png"), aligned)

    if affine is not None:
        print("affine matrix:\n", affine)
        np.savetxt(str(outdir / "standalone_affine.txt"), affine, fmt="%.8f")

    print("done")


if __name__ == "__main__":
    main()
