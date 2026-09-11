"""Run from the repository root: python -m example.saliency_demo --input IMAGE."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from seam_carving import resize, saliency_mask, spectral_residual


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and PPT saliency seam carving"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/saliency_demo"))
    parser.add_argument("--width-ratio", type=float, default=0.7)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--weight", type=float, default=1e11)
    args = parser.parse_args()
    if not 0 < args.width_ratio < 1:
        parser.error("--width-ratio must be between 0 and 1")
    src = np.array(Image.open(args.input).convert("RGB"))
    h, w = src.shape[:2]
    target = (max(1, round(w * args.width_ratio)), h)
    args.output.mkdir(parents=True, exist_ok=True)
    sal = spectral_residual(src)
    mask = saliency_mask(sal, args.threshold)
    results = {}
    times = {}
    for mode in ("backward", "saliency"):
        start = perf_counter()
        options = (
            dict(
                saliency_map=sal,
                saliency_threshold=args.threshold,
                saliency_weight=args.weight,
            )
            if mode == "saliency"
            else {}
        )
        results[mode] = resize(src, target, energy_mode=mode, **options)
        times[mode] = perf_counter() - start
        Image.fromarray(results[mode]).save(args.output / f"{mode}.png")
    Image.fromarray(src).save(args.output / "original.png")
    Image.fromarray((sal * 255).astype(np.uint8)).save(args.output / "saliency_map.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / "mask.png")
    overlay = src.copy()
    overlay[mask] = (0.55 * src[mask] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    Image.fromarray(overlay).save(args.output / "protection_overlay.png")
    panels = [
        ("Original", src),
        ("Protected regions (red)", overlay),
        ("Baseline: gradient", results["backward"]),
        ("PPT: gradient + saliency barrier", results["saliency"]),
    ]
    canvas = Image.new("RGB", (w * 2, (h + 40) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, arr) in enumerate(panels):
        x, y = (i % 2) * w, (i // 2) * (h + 40)
        draw.text((x + 12, y + 12), title, fill="black")
        canvas.paste(Image.fromarray(arr), (x + (w - arr.shape[1]) // 2, y + 40))
    canvas.save(args.output / "comparison.png")
    report = dict(
        input=str(args.input),
        original_size=[w, h],
        target_size=list(target),
        threshold=args.threshold,
        weight=args.weight,
        protected_fraction=float(mask.mean()),
        seconds=times,
    )
    (args.output / "run.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
