Saliency-Guided Seam Carving

A Python project exploring visual saliency for content-aware image resizing. The goal is to reduce foreground distortion when textured backgrounds receive higher gradient energy than smooth subjects.

## Original Work

- **Original algorithm:** Shai Avidan and Ariel Shamir, [Seam Carving for Content-Aware Image Resizing](https://www.merl.com/publications/TR2007-087), ACM SIGGRAPH / TOG, 2007.
- **Base implementation:** Jiahao Li's [li-plus/seam-carving](https://github.com/li-plus/seam-carving), distributed under the MIT License.

This repository extends the existing implementation; the original Seam Carving algorithm and baseline code are credited to their respective authors.

## Changes in This Project

- Completed the missing dynamic-programming code in the local baseline.
- Added spectral-residual saliency detection and a binary protection mask.
- Integrated a saliency-guided energy function: **E_new = E_gradient + λ × M_saliency**, with a default protection weight of **λ = 1e11**.
- Added `energy_mode="saliency"`, configurable protection parameters, and synchronized mask updates during resizing.
- Added a comparison demo, saliency-related tests, and implementation notes.

These extensions implement the optimization approach described in the project presentation. The existing backward and forward energy modes remain available.

## Quick Start

Run from the project root:

```bash
pip install -r requirements.txt
pip install -e .
python -m example.saliency_demo --input "image.png" --width-ratio 0.7 --threshold 0.2
```

The comparison image is saved to `outputs/saliency_demo/comparison.png`. A larger `--width-ratio` removes fewer seams; a lower `--threshold` protects more regions, including potentially unwanted background areas.

## Current Results

On the supplied test image, the modified method preserved the face better than the baseline, but the torso remained distorted. Spectral-residual saliency does not identify complete semantic objects, so full subject preservation is not guaranteed.

## License

The original MIT license and copyright notice are retained. See [LICENSE](LICENSE).
