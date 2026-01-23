from pathlib import Path
import argparse
import os
import sys
from typing import Optional, Tuple

import torch
from PIL import Image
import torchvision.transforms as T


def find_default_images(repo_root: Path) -> Optional[Tuple[Path, Path]]:
    src_dir = repo_root / "datasets" / "DIV2K_subset" / "DIV2K_train_HR"
    if not src_dir.exists():
        return None
    images = [
        p
        for p in sorted(src_dir.iterdir())
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if len(images) < 2:
        return None
    return images[0], images[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniGuard cover/secret embed demo")
    parser.add_argument("--cover", type=Path, default=None, help="Cover image path")
    parser.add_argument("--secret", type=Path, default=None, help="Secret image path")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: OmniGuard-main/checkpoint/model_checkpoint_01500.pt)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Resize size for cover/secret (default: 256)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: OmniGuard-main/output_real)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "OmniGuard-main"

    if args.checkpoint is None:
        args.checkpoint = root / "checkpoint" / "model_checkpoint_01500.pt"
    if args.out_dir is None:
        args.out_dir = root / "output_real"

    if args.cover is None or args.secret is None:
        defaults = find_default_images(repo_root)
        if defaults is None:
            raise SystemExit("No default DIV2K images found. Provide --cover/--secret.")
        args.cover, args.secret = defaults

    if not args.cover.exists() or not args.secret.exists():
        raise SystemExit("Cover or secret image not found.")
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for OmniGuard embed demo.")

    os.chdir(root)
    sys.path.insert(0, str(root))

    from model_invert import Model
    import modules.Unet_common as common

    transform = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
    ])

    cover = transform(Image.open(args.cover).convert("RGB")).unsqueeze(0).cuda()
    secret = transform(Image.open(args.secret).convert("RGB")).unsqueeze(0).cuda()

    model = Model().cuda().eval()
    state = torch.load(str(args.checkpoint), map_location="cpu")
    net_state = state["net"] if isinstance(state, dict) and "net" in state else state
    net_state = {k.replace("module.", ""): v for k, v in net_state.items()}
    model.load_state_dict(net_state, strict=False)

    with torch.no_grad():
        dwt = common.DWT()
        cover_input = dwt(cover)
        secret_input = dwt(secret)
        stego, _, _, _ = model(cover_input, secret_input)

    stego = stego.clamp(0, 1).cpu().squeeze(0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cover_out = args.out_dir / "cover.png"
    secret_out = args.out_dir / "secret.png"
    stego_out = args.out_dir / "stego.png"

    T.ToPILImage()(cover.cpu().squeeze(0)).save(cover_out)
    T.ToPILImage()(secret.cpu().squeeze(0)).save(secret_out)
    T.ToPILImage()(stego).save(stego_out)

    print(f"cover: {args.cover}")
    print(f"secret: {args.secret}")
    print(f"output: {stego_out}")


if __name__ == "__main__":
    main()
