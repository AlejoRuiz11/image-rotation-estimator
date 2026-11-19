# image-rotation-estimator

Estimate the rotation, scale and translation between two overhead images of the same area — for example,
a top-down render of a 3D model and an orthomosaic — **with a 95 % confidence interval**, and show the evidence.

> **Status:** work in progress.

## Why

Aligning two images is a classic, well-solved problem (OpenCV provides the building blocks; libraries such as
`imreg_dft` already estimate rotation with Fourier methods). This project does not claim a new algorithm. Its
focus is **verification**:

- report an uncertainty interval for every estimate;
- produce visual evidence (overlay and checkerboard) that a human can check at a glance.

## MVP scope

| Feature | Method |
|---|---|
| Feature-based registration | SIFT + ratio test + RANSAC (similarity) + least-squares refinement |
| Uncertainty | Bootstrap over inlier matches (95 % interval) |
| Visual evidence | Overlay and checkerboard of the aligned images |
| Quality warnings | Few inliers, high residual, implausible scale |

Possible later work: a Fourier cross-check, a Bayesian estimator, a coverage benchmark and learned matching.

## Planned usage

```bash
rotest compare moving.png reference.png --mask-reference reference_mask.png --out results/
```

## Angle convention

`angle_cw_deg` is the clockwise rotation, as displayed on screen, that must be applied to the **moving** image to
align it with the **reference** image. If the reference is north-up, `angle_cw_deg` is the compass heading of the
moving image's "up" direction. The convention is pinned by tests with known synthetic rotations.

## License

MIT. See [`LICENSE`](LICENSE).
