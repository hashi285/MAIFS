from pathlib import Path
import argparse
import sys
from typing import Optional
import numpy as np
from PIL import Image
import torch


def find_default_image(repo_root: Path) -> Optional[Path]:
    casia_tp = repo_root / "datasets" / "CASIA2_subset" / "Tp"
    if not casia_tp.exists():
        return None
    for path in sorted(casia_tp.iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            return path
    return None


def load_state(weights: Path):
    state = torch.load(str(weights), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="MVSS-Net single-image inference")
    parser.add_argument("--image", type=Path, default=None, help="Input image path")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Checkpoint path (default: MVSS-Net-master/ckpt/mvssnet_casia.pt)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: MVSS-Net-master/output_real)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    mvss_root = repo_root / "MVSS-Net-master"

    if args.weights is None:
        args.weights = mvss_root / "ckpt" / "mvssnet_casia.pt"
    if args.out_dir is None:
        args.out_dir = mvss_root / "output_real"
    if args.image is None:
        args.image = find_default_image(repo_root)

    if args.image is None or not args.image.exists():
        raise SystemExit("No input image found. Provide --image or stage CASIA2_subset.")
    if not args.weights.exists():
        raise SystemExit(f"Checkpoint not found: {args.weights}")

    sys.path.insert(0, str(mvss_root))
    from models.mvssnet import get_mvss
    from common.tools import inference_single

    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img)

    model = get_mvss(pretrained_base=False, sobel=True, n_input=3, constrain=True)
    model.load_state_dict(load_state(args.weights), strict=True)
    model.cuda()

    pred, score = inference_single(img_np, model)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.image.stem}_pred.png"
    Image.fromarray(pred).save(out_path)

    print(f"input: {args.image}")
    print(f"output: {out_path}")
    print(f"max_score: {score}")


if __name__ == "__main__":
    main()
