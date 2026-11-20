"""Local Streamlit app (optional `ui` extra): a thin layer over the same API as the CLI.

The calculation helpers (`run_pipeline`, `sign_test`, `overall_fraction`) never touch `st`, so
they can be tested without a browser; only `_app` draws widgets. Uploads and default outputs go
to a temporary directory, never to the repository.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from matplotlib.figure import Figure
from numpy.typing import NDArray

from rotest.geometry import normalize_angle
from rotest.io import DEFAULT_WORK_SIZE, Prepared, prepare
from rotest.register import (
    FLANN_CHECKS,
    FLANN_TREES,
    RATIO,
    RegistrationError,
    Result,
    default_threads,
    inlier_residuals,
    register,
    set_threads,
)
from rotest.report import TILE_PX, _checkerboard, quality_warnings, warp_moving, write_report
from rotest.uncertainty import (
    MIN_REGION_INLIERS,
    N_BOOT,
    SpatialCheck,
    angle_uncertainty,
    bootstrap_angles,
    ci95_from_angles,
    spatial_check,
)

SIGN_TEST_FLOOR_DEG = 0.2  # reported uncertainty can fall short ~2x with real data
VIEW_MAX_SIDE = 1600  # longest side of every image shown in the browser
MAX_DRAWN_MATCHES = 300  # inlier lines drawn at most, sampled with a fixed seed
OUTPUT_FILES = ("results.json", "summary.md", "overlay.png", "checkerboard.png")

STAGES: tuple[str, ...] = (
    "load",
    "detect_moving",
    "detect_reference",
    "match",
    "ransac",
    "refine",
    "bootstrap",
    "spatial",
    "report",
)
STAGE_LABELS: dict[str, str] = {
    "load": "Loading images",
    "detect_moving": "Detecting features",
    "detect_reference": "Detecting features",
    "match": "Matching",
    "ransac": "RANSAC",
    "refine": "Refining",
    "bootstrap": "Bootstrap",
    "spatial": "Checking spatial consistency",
    "report": "Writing reports",
}

ProgressFn = Callable[[str, float], None]


def overall_fraction(stage: str, fraction: float) -> float:
    """Map (stage, fraction within stage) to overall progress in [0, 1]."""
    return (STAGES.index(stage) + fraction) / len(STAGES)


@dataclass(frozen=True)
class Run:
    """Everything one pipeline run produces, kept in the session so widgets do not recompute."""

    moving: Prepared
    reference: Prepared
    result: Result
    ci: tuple[float, float]
    spatial: SpatialCheck
    uncertainty: float
    warnings: list[str]
    boot_angles: NDArray[np.float64]


def run_pipeline(
    moving_path: str | Path,
    reference_path: str | Path,
    out_dir: str | Path | None = None,
    mask_moving: str | Path | None = None,
    mask_reference: str | Path | None = None,
    work_size: int = DEFAULT_WORK_SIZE,
    threads: int | None = None,
    progress: ProgressFn | None = None,
) -> Run:
    """Same calls and defaults as `rotest compare`; without `out_dir` no report is written."""

    def emit(stage: str, fraction: float) -> None:
        if progress:
            progress(stage, fraction)

    set_threads(threads or default_threads())
    emit("load", 0.0)
    moving = prepare(moving_path, mask_moving, work_size)
    reference = prepare(reference_path, mask_reference, work_size)
    emit("load", 1.0)
    result = register(moving, reference, progress=progress)
    boot_angles = bootstrap_angles(result, progress=progress)
    ci = ci95_from_angles(boot_angles)
    spatial = spatial_check(result, progress=progress)
    if out_dir is not None:
        emit("report", 0.0)
        warnings = write_report(out_dir, moving, reference, result, ci, spatial)
        emit("report", 1.0)
    else:
        warnings = quality_warnings(result, spatial)
    uncertainty = angle_uncertainty(ci, spatial)
    return Run(moving, reference, result, ci, spatial, uncertainty, warnings, boot_angles)


@dataclass(frozen=True)
class SignTest:
    """Outcome of the rotated-copy sign check."""

    difference_deg: float
    expected_deg: float
    tolerance_deg: float
    tolerance_source: str
    passed: bool


def sign_test(
    angle_deg: float,
    rotated_angle_deg: float,
    expected_change_deg: float,
    u_deg: float,
    rotated_u_deg: float,
) -> SignTest:
    """Check that rotating the moving image changed the estimate by the expected amount."""
    difference = normalize_angle(rotated_angle_deg - angle_deg)
    u_sum = u_deg + rotated_u_deg
    tolerance = max(SIGN_TEST_FLOOR_DEG, u_sum)
    source = "floor" if SIGN_TEST_FLOOR_DEG >= u_sum else "U1 + U2"
    passed = abs(difference - expected_change_deg) <= tolerance
    return SignTest(difference, expected_change_deg, tolerance, source, passed)


def shrink_view(
    img: NDArray[np.uint8], max_side: int = VIEW_MAX_SIDE
) -> tuple[NDArray[np.uint8], float]:
    """Reduce `img` so its longest side is <= `max_side`; return it with the factor applied."""
    factor = min(1.0, max_side / max(img.shape[:2]))
    if factor == 1.0:
        return img, 1.0
    size = (max(1, round(img.shape[1] * factor)), max(1, round(img.shape[0] * factor)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), factor


@dataclass(frozen=True)
class Views:
    """Reduced images computed once per run; points scale with the matching factor."""

    moving: NDArray[np.uint8]
    reference: NDArray[np.uint8]
    aligned: NDArray[np.uint8]  # moving warped onto the reference grid
    moving_factor: float
    reference_factor: float


def build_views(run: Run) -> Views:
    """Reduce the working images and the aligned moving image for display."""
    moving, f_m = shrink_view(run.moving.image)
    reference, f_r = shrink_view(run.reference.image)
    aligned, _ = shrink_view(warp_moving(run.moving, run.reference, run.result))
    return Views(moving, reference, aligned, f_m, f_r)


def select_matches(
    n: int, max_matches: int = MAX_DRAWN_MATCHES, seed: int = 0
) -> NDArray[np.intp]:
    """Sorted indices of at most `max_matches` of `n` pairs, sampled with a fixed seed."""
    if n <= max_matches:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=max_matches, replace=False))


def draw_matches(views: Views, result: Result) -> NDArray[np.uint8]:
    """Moving (left) and reference (right) side by side with a line per sampled inlier (BGR)."""
    h = max(views.moving.shape[0], views.reference.shape[0])
    w_m = views.moving.shape[1]
    canvas = np.zeros((h, w_m + views.reference.shape[1], 3), np.uint8)
    canvas[: views.moving.shape[0], :w_m] = views.moving[..., None]
    canvas[: views.reference.shape[0], w_m:] = views.reference[..., None]
    for i in select_matches(len(result.src_pts)):
        p = (result.src_pts[i] * views.moving_factor).round().astype(int)
        q = (result.dst_pts[i] * views.reference_factor).round().astype(int) + (w_m, 0)
        cv2.line(canvas, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (0, 220, 0), 2, cv2.LINE_AA)
    return shrink_view(canvas)[0]


def blend(
    reference: NDArray[np.uint8], aligned: NDArray[np.uint8], alpha: float
) -> NDArray[np.uint8]:
    """Mix the images: alpha 0 gives the reference, alpha 1 the aligned moving image."""
    return cv2.addWeighted(reference, 1.0 - alpha, aligned, alpha, 0)


def split_view(
    reference: NDArray[np.uint8], aligned: NDArray[np.uint8], position: float
) -> NDArray[np.uint8]:
    """Reference left of a vertical cut at `position` (0..1 of the width), aligned right of it."""
    out = aligned.copy()
    cut = round(position * out.shape[1])
    out[:, :cut] = reference[:, :cut]
    return out


def checkerboard_view(views: Views) -> NDArray[np.uint8]:
    """Checkerboard of reference and aligned moving, with tiles scaled like the images."""
    tile = max(1, round(TILE_PX * views.reference_factor))
    return _checkerboard(views.reference, views.aligned, tile)


def histogram_figure(
    boot_angles: NDArray[np.float64], ci: tuple[float, float], angle_deg: float
) -> Figure:
    """Histogram of the bootstrap angles with the estimate and the 95% interval marked."""
    fig = Figure(figsize=(6, 3))
    ax = fig.subplots()
    ax.hist(boot_angles, bins=40, color="0.7")
    ax.axvline(angle_deg, color="k", label="estimate")
    ax.axvline(ci[0], color="r", linestyle="--", label="95% interval")
    ax.axvline(ci[1], color="r", linestyle="--")
    ax.set_xlabel("clockwise angle (deg)")
    ax.set_ylabel("resamples")
    ax.legend()
    fig.tight_layout()
    return fig


def residual_figure(views: Views, result: Result) -> Figure:
    """Inliers over the reference, colored by their error in reference working pixels."""
    fig = Figure(figsize=(7, 6))
    ax = fig.subplots()
    ax.imshow(views.reference, cmap="gray")
    pts = result.dst_pts * views.reference_factor
    scatter = ax.scatter(pts[:, 0], pts[:, 1], c=inlier_residuals(result), cmap="viridis", s=14)
    fig.colorbar(scatter, ax=ax, label="residual (px)")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def read_outputs(out_dir: str | Path) -> dict[str, bytes]:
    """Bytes of the report files in `out_dir`, for the download buttons."""
    return {name: (Path(out_dir) / name).read_bytes() for name in OUTPUT_FILES}


def _save_upload(directory: Path, label: str, upload: object) -> Path:
    """Write an uploaded file into `directory`, keeping its extension; return the path."""
    suffix = Path(str(upload.name)).suffix  # type: ignore[attr-defined]
    path = directory / f"{label}{suffix}"
    path.write_bytes(upload.getvalue())  # type: ignore[attr-defined]
    return path


def _progress_callback(bar: object, text: object) -> ProgressFn:
    def update(stage: str, fraction: float) -> None:
        bar.progress(min(1.0, overall_fraction(stage, fraction)))  # type: ignore[attr-defined]
        text.text(STAGE_LABELS[stage])  # type: ignore[attr-defined]

    return update


def _app() -> None:
    st.set_page_config(page_title="rotest", layout="wide")
    st.title("Image rotation estimator")
    st.caption("Runs locally: nothing is uploaded anywhere.")

    col_a, col_b = st.columns(2)
    moving_up = col_a.file_uploader("Moving image", key="moving")
    reference_up = col_b.file_uploader("Reference image", key="reference")
    mask_m_up = col_a.file_uploader("Moving mask (optional)", key="mask_moving")
    mask_r_up = col_b.file_uploader("Reference mask (optional)", key="mask_reference")
    rotated_up = st.file_uploader("Rotated test moving image (optional, for the sign test)", key="rotated")
    expected = st.number_input(
        "Expected angle change (deg)",
        value=10.0,
        help=(
            "If the test image was rotated counter-clockwise on screen by 10 deg, the estimated "
            "clockwise angle must increase by 10 deg (e.g. 0.18 -> 10.18)."
        ),
    )

    with st.expander("Advanced"):
        work_size = int(st.number_input("Work size (px)", min_value=64, value=DEFAULT_WORK_SIZE))
        threads = int(st.number_input("Threads", min_value=1, value=default_threads()))
        st.markdown(
            "Fixed values (not editable): "
            f"ratio test {RATIO}, FLANN trees {FLANN_TREES}, FLANN checks {FLANN_CHECKS}, "
            f"bootstrap resamples {N_BOOT}, min. inliers per region {MIN_REGION_INLIERS}, "
            f"sign-test floor {SIGN_TEST_FLOOR_DEG} deg."
        )

    if "workdir" not in st.session_state:
        st.session_state["workdir"] = tempfile.mkdtemp(prefix="rotest-")
    workdir = Path(st.session_state["workdir"])
    out_text = st.text_input("Output folder", value=str(workdir / "out"))

    if st.button("Run", disabled=moving_up is None or reference_up is None):
        bar, text = st.progress(0.0), st.empty()
        callback = _progress_callback(bar, text)
        try:
            mov = _save_upload(workdir, "moving", moving_up)
            ref = _save_upload(workdir, "reference", reference_up)
            m_mask = _save_upload(workdir, "mask_moving", mask_m_up) if mask_m_up else None
            r_mask = _save_upload(workdir, "mask_reference", mask_r_up) if mask_r_up else None
            common = {
                "mask_moving": m_mask,
                "mask_reference": r_mask,
                "work_size": work_size,
                "threads": threads,
            }
            run = run_pipeline(mov, ref, out_text, progress=callback, **common)  # type: ignore[arg-type]
            rotated = None
            if rotated_up is not None:
                rot = _save_upload(workdir, "rotated", rotated_up)
                rotated = run_pipeline(rot, ref, None, progress=callback, **common)  # type: ignore[arg-type]
            st.session_state["run"] = run
            st.session_state["views"] = build_views(run)
            st.session_state["downloads"] = read_outputs(out_text)
            st.session_state["rotated_run"] = rotated
            st.session_state["out_dir"] = out_text
        except RegistrationError as exc:
            st.error(f"Registration failed, not enough reliable matches: {exc}")
        except (ValueError, OSError) as exc:
            st.error(f"Error: {exc}")
        finally:
            bar.empty()
            text.empty()

    run: Run | None = st.session_state.get("run")
    if run is None:
        return
    views: Views = st.session_state["views"]
    _result_card(run)
    rotated_run: Run | None = st.session_state.get("rotated_run")
    if rotated_run is not None:
        _sign_card(run, rotated_run, float(expected))

    st.header("Matches")
    st.caption("Inlier pairs between the moving image (left) and the reference (right).")
    st.image(draw_matches(views, run.result), channels="BGR")

    st.header("Comparison")
    st.caption("Moving image aligned onto the reference: blend them or compare them across a cut.")
    mode = st.radio("Mode", ["Blend", "Split"], horizontal=True)
    if mode == "Blend":
        alpha = st.slider("Blend (0 = reference, 1 = moving)", 0.0, 1.0, 0.5)
        st.image(blend(views.reference, views.aligned, alpha))
    else:
        cut = st.slider("Cut position", 0.0, 1.0, 0.5)
        st.image(split_view(views.reference, views.aligned, cut))
    st.caption("Checkerboard: tiles alternate between the reference and the aligned moving image.")
    st.image(checkerboard_view(views))

    st.header("Bootstrap angles")
    st.caption("Angles of the resampled fits; dashed lines mark the 95% interval.")
    st.pyplot(histogram_figure(run.boot_angles, run.ci, run.result.angle_cw_deg))

    st.header("Residual map")
    st.caption("Each inlier on the reference, colored by its distance to the fitted position.")
    st.pyplot(residual_figure(views, run.result))

    st.header("Downloads")
    st.caption(f"Files written to {st.session_state['out_dir']}.")
    for name, data in st.session_state["downloads"].items():
        st.download_button(name, data, file_name=name)


def _result_card(run: Run) -> None:
    """Main card: angle, interval, uncertainty, quality numbers, warnings, region table."""
    r = run.result
    st.header(f"Angle: {r.angle_cw_deg:.3f} deg clockwise")
    st.caption(
        "Clockwise rotation, as seen on screen, that aligns the moving image with the reference; "
        "the uncertainty adds the bootstrap half-width and the spatial deviation."
    )
    cols = st.columns(5)
    cols[0].metric("95% interval (deg)", f"{run.ci[0]:.3f} to {run.ci[1]:.3f}")
    cols[1].metric("Uncertainty (deg)", f"+-{run.uncertainty:.3f}")
    cols[2].metric("Scale", f"{r.scale:.4f}")
    cols[3].metric("Inliers / matches", f"{r.inliers} / {r.matches}")
    cols[4].metric("Residual RMS (px)", f"{r.residual_rms_px:.2f}")
    for warning in run.warnings:
        st.warning(warning)
    st.caption("Spatial consistency: the inliers are fitted per quadrant of the moving image.")
    st.table(
        [
            {
                "Region": reg.name,
                "Inliers": reg.inliers,
                "Angle (deg)": "n/a" if reg.angle_cw_deg is None else f"{reg.angle_cw_deg:.3f}",
                "95% interval (deg)": (
                    "n/a"
                    if reg.angle_ci95 is None
                    else f"{reg.angle_ci95[0]:.3f} to {reg.angle_ci95[1]:.3f}"
                ),
                "Scale": "n/a" if reg.scale is None else f"{reg.scale:.4f}",
            }
            for reg in run.spatial.regions
        ]
    )
    dev = run.spatial.angle_dev_deg
    st.write(f"Spatial deviation beyond noise: {'n/a' if dev is None else f'{dev:.3f} deg'}.")


def _sign_card(run: Run, rotated_run: Run, expected: float) -> None:
    """Sign-test card: difference, expected change, tolerance and its origin."""
    sign = sign_test(
        run.result.angle_cw_deg,
        rotated_run.result.angle_cw_deg,
        expected,
        run.uncertainty,
        rotated_run.uncertainty,
    )
    origin = (
        f"floor {SIGN_TEST_FLOOR_DEG} deg"
        if sign.tolerance_source == "floor"
        else f"U1 + U2 = {sign.tolerance_deg:.3f} deg"
    )
    st.subheader("Sign test")
    st.caption(
        f"The tolerance is the larger of {SIGN_TEST_FLOOR_DEG} deg and the sum of both total "
        "uncertainties, because with real data the reported uncertainty can fall short (~2x)."
    )
    message = (
        f"{'PASS' if sign.passed else 'FAIL'}: measured change {sign.difference_deg:.4f} deg, "
        f"expected {sign.expected_deg:.4f} deg, tolerance {sign.tolerance_deg:.3f} deg ({origin})."
    )
    (st.success if sign.passed else st.error)(message)


if __name__ == "__main__":
    _app()
