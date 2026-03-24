"""Reusable video production pipeline for the MuJoCo Robotics Lab series.

This module standardizes the three-phase demo workflow used by every lab:

1. Animated metrics presentation with Matplotlib.
2. Native MuJoCo simulation recording with overlays and slow motion.
3. ffmpeg composition into a final H.264 demo artifact.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
VIDEO_CODEC = "libx264"
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
CARD_DURATION_SEC = 2.0

BACKGROUND_COLOR = "#08111f"
PANEL_COLOR = "#101d33"
GRID_COLOR = "#26324a"
TEXT_COLOR = "#e8edf6"
SUBTEXT_COLOR = "#9ba7bb"
ACCENT_COLORS = (
    "#58c4dd",
    "#ff9e66",
    "#9bde7e",
    "#ffd166",
    "#c792ea",
    "#82aaff",
)

COMMON_TRACE_SITE_NAMES = (
    "2f85_pinch",
    "pinch_link",
    "ee_site",
    "attachment_site",
)

DEFAULT_CAMERA_PRESETS: dict[str, dict[str, float | list[float]]] = {
    "fixed_top": {
        "lookat": [0.55, 0.00, 0.42],
        "distance": 1.40,
        "elevation": -88.0,
        "azimuth": 90.0,
    },
    "orbit_45": {
        "lookat": [0.55, 0.00, 0.42],
        "distance": 1.55,
        "elevation": -35.0,
        "azimuth": 130.0,
    },
}


def _lab_file_prefix(lab_name: str) -> str:
    """Create a stable file prefix from the lab name."""
    match = re.search(r"lab\W*(\d+)", lab_name, flags=re.IGNORECASE)
    if match:
        return f"lab{int(match.group(1))}"

    slug = re.sub(r"[^a-z0-9]+", "_", lab_name.lower()).strip("_")
    return slug or "lab"


def _ease_in_out(t: float) -> float:
    """Smooth cubic easing."""
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def _as_rgb_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Validate and normalize a frame for ffmpeg writing."""
    array = np.asarray(frame)
    if array.shape != (height, width, 3):
        raise ValueError(
            f"Expected frame shape {(height, width, 3)}, got {array.shape}."
        )
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _matplotlib_font_path() -> Path:
    """Return a reliable font shipped with Matplotlib."""
    return Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable font for overlays and cards."""
    font_path = _matplotlib_font_path()
    if bold:
        bold_path = font_path.with_name("DejaVuSans-Bold.ttf")
        if bold_path.exists():
            return ImageFont.truetype(str(bold_path), size=size)
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _make_gradient_background(width: int, height: int) -> np.ndarray:
    """Create a subtle dark gradient background."""
    y = np.linspace(0.0, 1.0, height, dtype=float)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=float)[None, :]
    gradient = 0.55 * y + 0.45 * x

    top = np.array([8, 17, 31], dtype=float)
    bottom = np.array([14, 36, 60], dtype=float)
    accent = np.array([12, 88, 110], dtype=float)

    image = top + gradient[..., None] * (bottom - top)
    image += (0.12 * np.sin(2.0 * math.pi * x))[..., None] * accent
    return np.clip(image, 0, 255).astype(np.uint8)


def _make_card_frame(
    title: str,
    subtitle: str,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    footer: str | None = None,
) -> np.ndarray:
    """Render a polished title or end card."""
    image = Image.fromarray(_make_gradient_background(width, height))
    draw = ImageDraw.Draw(image, "RGBA")

    title_font = _load_font(72, bold=True)
    subtitle_font = _load_font(34)
    footer_font = _load_font(24)

    draw.rounded_rectangle(
        (120, 150, width - 120, height - 150),
        radius=36,
        fill=(6, 15, 27, 180),
        outline=(88, 196, 221, 180),
        width=3,
    )
    draw.rounded_rectangle(
        (160, 220, 460, 252),
        radius=12,
        fill=(88, 196, 221, 255),
    )

    title_lines = textwrap.wrap(title, width=28) or [title]
    y = 330
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        text_w = bbox[2] - bbox[0]
        draw.text(
            ((width - text_w) / 2, y),
            line,
            font=title_font,
            fill=TEXT_COLOR,
        )
        y += 92

    subtitle_lines = textwrap.wrap(subtitle, width=42) or [subtitle]
    y += 28
    for line in subtitle_lines:
        bbox = draw.textbbox((0, 0), line, font=subtitle_font)
        text_w = bbox[2] - bbox[0]
        draw.text(
            ((width - text_w) / 2, y),
            line,
            font=subtitle_font,
            fill=SUBTEXT_COLOR,
        )
        y += 48

    if footer:
        bbox = draw.textbbox((0, 0), footer, font=footer_font)
        text_w = bbox[2] - bbox[0]
        draw.text(
            ((width - text_w) / 2, height - 120),
            footer,
            font=footer_font,
            fill="#7ed3cf",
        )

    return np.asarray(image, dtype=np.uint8)


class _StreamingVideoWriter:
    """Streaming ffmpeg-backed RGB video writer."""

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps

        self._generator = imageio_ffmpeg.write_frames(
            str(self.output_path),
            (self.width, self.height),
            fps=float(self.fps),
            codec=VIDEO_CODEC,
            pix_fmt_in="rgb24",
            output_params=[
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
            ],
        )
        self._generator.send(None)

    def write(self, frame: np.ndarray) -> None:
        """Write one RGB frame."""
        self._generator.send(_as_rgb_frame(frame, self.width, self.height))

    def close(self) -> None:
        """Finalize the ffmpeg process."""
        if self._generator is not None:
            self._generator.close()
            self._generator = None

    def __enter__(self) -> _StreamingVideoWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _figure_to_frame(fig: plt.Figure) -> np.ndarray:
    """Convert a Matplotlib figure canvas to an RGB frame."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, :3].copy())


def _series_from_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize plot data to a list of series dictionaries."""
    if "series" in spec:
        return [dict(entry) for entry in spec["series"]]

    data = spec["data"]
    default_color = spec.get("color")
    label = spec.get("label")
    ptype = spec.get("type", "line")

    if ptype == "bar":
        labels, values = data
        return [
            {
                "type": "bar",
                "labels": list(labels),
                "values": np.asarray(values, dtype=float),
                "label": label,
                "color": default_color,
            }
        ]

    if (
        isinstance(data, list)
        and data
        and isinstance(data[0], (dict, tuple, list))
        and ptype in {"line", "scatter"}
    ):
        series: list[dict[str, Any]] = []
        for idx, entry in enumerate(data):
            if isinstance(entry, Mapping):
                series.append(dict(entry))
                series[-1].setdefault("type", ptype)
            elif len(entry) == 3:
                x, y, series_label = entry
                series.append(
                    {
                        "type": ptype,
                        "x": np.asarray(x, dtype=float),
                        "y": np.asarray(y, dtype=float),
                        "label": series_label,
                    }
                )
            else:
                x, y = entry
                series.append(
                    {
                        "type": ptype,
                        "x": np.asarray(x, dtype=float),
                        "y": np.asarray(y, dtype=float),
                        "label": f"Series {idx + 1}",
                    }
                )
        return series

    x, y = data
    return [
        {
            "type": ptype,
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
            "label": label,
            "color": default_color,
        }
    ]


def _apply_axis_style(ax: plt.Axes) -> None:
    """Apply the shared dark theme to an axis."""
    ax.set_facecolor(PANEL_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.45)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(SUBTEXT_COLOR)
    ax.yaxis.label.set_color(SUBTEXT_COLOR)


def _draw_kpi_panel(fig: plt.Figure, kpi_overlay: Mapping[str, str]) -> None:
    """Add the right-side KPI panel to the metrics figure."""
    x0 = 0.80
    fig.text(
        x0,
        0.91,
        "Key Metrics",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="left",
        va="center",
    )

    for idx, (key, value) in enumerate(kpi_overlay.items()):
        y = 0.84 - idx * 0.105
        fig.text(
            x0,
            y,
            key,
            fontsize=11,
            color=SUBTEXT_COLOR,
            ha="left",
            va="center",
        )
        fig.text(
            x0,
            y - 0.035,
            value,
            fontsize=18,
            fontweight="bold",
            color=ACCENT_COLORS[idx % len(ACCENT_COLORS)],
            ha="left",
            va="center",
        )


def _resolve_trace_site_id(model: mujoco.MjModel, site_name: str | None) -> int | None:
    """Resolve the site used for the executed trajectory trace."""
    names = [site_name] if site_name else list(COMMON_TRACE_SITE_NAMES)
    for name in names:
        if not name:
            continue
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id != -1:
            return int(site_id)
    return None


def _camera_from_name(model: mujoco.MjModel, camera_name: str) -> mujoco.MjvCamera:
    """Create a camera from a named MuJoCo camera or a shared preset."""
    camera = mujoco.MjvCamera()
    if camera_name in DEFAULT_CAMERA_PRESETS:
        preset = DEFAULT_CAMERA_PRESETS[camera_name]
        _apply_camera_state(camera, preset)
        return camera

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id == -1:
        _apply_camera_state(camera, DEFAULT_CAMERA_PRESETS["fixed_top"])
        return camera

    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = camera_id
    return camera


def _apply_camera_state(
    camera: mujoco.MjvCamera,
    state: Mapping[str, Any],
) -> None:
    """Apply a free-camera pose."""
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(state.get("lookat", [0.55, 0.0, 0.42]), dtype=float)
    camera.distance = float(state.get("distance", 1.50))
    camera.elevation = float(state.get("elevation", -35.0))
    camera.azimuth = float(state.get("azimuth", 135.0))


def _normalize_overlay_lines(
    overlay: Mapping[str, Any] | Sequence[str] | str | None,
) -> list[str]:
    """Normalize overlay content to display lines."""
    if overlay is None:
        return []
    if isinstance(overlay, str):
        return [overlay]
    if isinstance(overlay, Mapping):
        return [f"{key}: {value}" for key, value in overlay.items()]
    return [str(entry) for entry in overlay]


def _draw_text_overlay(
    frame: np.ndarray,
    lines: Sequence[str],
    *,
    margin: int = 48,
) -> np.ndarray:
    """Draw a status box on top of a rendered frame."""
    if not lines:
        return frame

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _load_font(24, bold=True)
    text_font = _load_font(24)

    wrapped_lines: list[str] = []
    for idx, line in enumerate(lines):
        width = 32 if idx == 0 else 38
        wrapped = textwrap.wrap(line, width=width) or [line]
        wrapped_lines.extend(wrapped)

    line_heights = []
    max_width = 0
    fonts = []
    for idx, line in enumerate(wrapped_lines):
        font = title_font if idx == 0 else text_font
        fonts.append(font)
        bbox = draw.textbbox((0, 0), line, font=font)
        max_width = max(max_width, bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    box_width = max_width + 42
    box_height = sum(line_heights) + 18 * len(line_heights) + 26
    x0 = margin
    y0 = margin

    draw.rounded_rectangle(
        (x0, y0, x0 + box_width, y0 + box_height),
        radius=18,
        fill=(6, 15, 27, 170),
        outline=(126, 211, 207, 170),
        width=2,
    )

    y = y0 + 16
    for line, font in zip(wrapped_lines, fonts):
        draw.text((x0 + 20, y), line, font=font, fill=TEXT_COLOR)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1]) + 18

    return np.asarray(image, dtype=np.uint8)


def _add_geom_capacity(scene: mujoco.MjvScene) -> bool:
    """Return whether another custom geom can be appended to the scene."""
    return scene.ngeom < scene.maxgeom


def _add_sphere(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    """Append a visual-only sphere to the current scene."""
    if not _add_geom_capacity(scene):
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0], dtype=float),
        pos=np.asarray(position, dtype=float),
        mat=np.eye(3).reshape(-1),
        rgba=np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


def _add_segment(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    """Append a connector segment to the current scene."""
    if not _add_geom_capacity(scene):
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3, dtype=float),
        pos=np.zeros(3, dtype=float),
        mat=np.eye(3).reshape(-1),
        rgba=np.asarray(rgba, dtype=float),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )
    scene.ngeom += 1


def add_polyline_to_scene(
    scene: mujoco.MjvScene,
    points: np.ndarray | Sequence[Sequence[float]],
    *,
    radius: float = 0.003,
    rgba: tuple[float, float, float, float] = (0.20, 0.90, 0.35, 0.85),
    stride: int = 1,
    max_segments: int = 220,
) -> None:
    """Draw a polyline in the current MuJoCo scene."""
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0:
        return
    if array.shape[0] == 1:
        _add_sphere(scene, array[0], radius * 1.3, rgba)
        return

    stride = max(1, int(stride))
    points_to_draw = array[::stride]
    if points_to_draw.shape[0] < 2:
        points_to_draw = array[:2]

    if points_to_draw.shape[0] - 1 > max_segments:
        subsample = int(math.ceil((points_to_draw.shape[0] - 1) / max_segments))
        points_to_draw = points_to_draw[::subsample]
        if not np.allclose(points_to_draw[-1], array[-1]):
            points_to_draw = np.vstack([points_to_draw, array[-1]])

    for start, end in zip(points_to_draw[:-1], points_to_draw[1:]):
        _add_segment(scene, start, end, radius, rgba)


def add_line_to_scene(
    scene: mujoco.MjvScene,
    points: np.ndarray | Sequence[Sequence[float]],
    *,
    size: float = 0.003,
    rgba: tuple[float, float, float, float] = (0.20, 0.90, 0.35, 0.85),
    stride: int = 1,
) -> None:
    """Compatibility wrapper for older demo scripts."""
    add_polyline_to_scene(scene, points, radius=size, rgba=rgba, stride=stride)


def add_marker_to_scene(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    *,
    size: float = 0.014,
    rgba: tuple[float, float, float, float] = (0.92, 0.28, 0.22, 1.0),
) -> None:
    """Add a marker sphere to the current scene."""
    _add_sphere(scene, np.asarray(position, dtype=float), size, rgba)


class LabVideoProducer:
    """Reusable three-phase demo video producer."""

    def __init__(self, lab_name: str, output_dir: Path) -> None:
        self.lab_name = lab_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_prefix = _lab_file_prefix(lab_name)
        self.last_simulation_trace: np.ndarray | None = None

    def _metrics_path(self) -> Path:
        return self.output_dir / f"{self.file_prefix}_metrics.mp4"

    def _simulation_path(self) -> Path:
        return self.output_dir / f"{self.file_prefix}_simulation.mp4"

    def _demo_path(self) -> Path:
        return self.output_dir / f"{self.file_prefix}_demo.mp4"

    def _build_metrics_figure(
        self,
        plots: list[dict[str, Any]],
        kpi_overlay: Mapping[str, str],
        title_text: str,
    ) -> tuple[plt.Figure, list[Callable[[float], None]]]:
        """Build the metrics figure and its animation callbacks."""
        fig = plt.figure(figsize=(VIDEO_WIDTH / 100, VIDEO_HEIGHT / 100), dpi=100)
        fig.patch.set_facecolor(BACKGROUND_COLOR)

        fig.text(
            0.07,
            0.95,
            self.lab_name,
            fontsize=16,
            fontweight="bold",
            color="#7ed3cf",
            ha="left",
            va="center",
        )
        fig.text(
            0.07,
            0.905,
            title_text,
            fontsize=30,
            fontweight="bold",
            color=TEXT_COLOR,
            ha="left",
            va="center",
        )
        fig.text(
            0.07,
            0.865,
            "Animated planning and execution metrics",
            fontsize=12,
            color=SUBTEXT_COLOR,
            ha="left",
            va="center",
        )

        _draw_kpi_panel(fig, kpi_overlay)

        rows = 2
        cols = 2
        grid = fig.add_gridspec(
            rows,
            cols,
            left=0.06,
            right=0.76,
            bottom=0.08,
            top=0.80,
            wspace=0.20,
            hspace=0.24,
        )

        plot_updates: list[Callable[[float], None]] = []

        for idx, spec in enumerate(plots[:4]):
            ax = fig.add_subplot(grid[idx // cols, idx % cols])
            _apply_axis_style(ax)
            ax.set_title(spec.get("title", ""), fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel(spec.get("xlabel", ""), fontsize=10)
            ax.set_ylabel(spec.get("ylabel", ""), fontsize=10)

            series = _series_from_spec(spec)
            overlays = [dict(entry) for entry in spec.get("overlays", [])]
            plot_type = spec.get("type", "line")

            x_values: list[np.ndarray] = []
            y_values: list[np.ndarray] = []
            artists: list[tuple[str, Any, dict[str, Any]]] = []

            for series_idx, series_spec in enumerate(series):
                color = series_spec.get(
                    "color",
                    ACCENT_COLORS[(idx + series_idx) % len(ACCENT_COLORS)],
                )
                if series_spec["type"] == "bar":
                    labels = series_spec["labels"]
                    values = np.asarray(series_spec["values"], dtype=float)
                    bars = ax.bar(
                        np.arange(len(labels)),
                        np.zeros_like(values),
                        color=color,
                        alpha=0.85,
                        linewidth=0.8,
                        edgecolor="#dce4f0",
                    )
                    ax.set_xticks(np.arange(len(labels)))
                    ax.set_xticklabels(labels, rotation=24, ha="right")
                    ax.set_ylim(
                        0.0,
                        float(np.max(values) * 1.18 + 1e-9),
                    )
                    artists.append(("bar", bars, {"values": values}))
                elif series_spec["type"] == "scatter":
                    x = np.asarray(series_spec["x"], dtype=float)
                    y = np.asarray(series_spec["y"], dtype=float)
                    scatter = ax.scatter([], [], s=10, color=color, alpha=0.75)
                    x_values.append(x)
                    y_values.append(y)
                    artists.append(("scatter", scatter, {"x": x, "y": y}))
                else:
                    x = np.asarray(series_spec["x"], dtype=float)
                    y = np.asarray(series_spec["y"], dtype=float)
                    line, = ax.plot(
                        [],
                        [],
                        color=color,
                        linewidth=series_spec.get("linewidth", 2.4),
                        label=series_spec.get("label"),
                    )
                    x_values.append(x)
                    y_values.append(y)
                    artists.append(("line", line, {"x": x, "y": y}))

            if x_values:
                all_x = np.concatenate(x_values)
                x_margin = max(1e-6, 0.05 * float(all_x.max() - all_x.min() + 1e-9))
                ax.set_xlim(float(all_x.min() - x_margin), float(all_x.max() + x_margin))
            if y_values:
                all_y = np.concatenate(y_values)
                y_margin = max(1e-6, 0.10 * float(all_y.max() - all_y.min() + 1e-9))
                ax.set_ylim(float(all_y.min() - y_margin), float(all_y.max() + y_margin))

            if plot_type == "scatter":
                ax.set_aspect("auto")

            overlay_artists: list[tuple[str, Any, dict[str, Any]]] = []
            for overlay_idx, overlay_spec in enumerate(overlays):
                overlay_type = overlay_spec.get("type", "line")
                color = overlay_spec.get(
                    "color",
                    ACCENT_COLORS[(idx + overlay_idx + 1) % len(ACCENT_COLORS)],
                )
                if overlay_type == "scatter":
                    x = np.asarray(overlay_spec["x"], dtype=float)
                    y = np.asarray(overlay_spec["y"], dtype=float)
                    scatter = ax.scatter([], [], s=18, color=color, alpha=0.95)
                    overlay_artists.append(
                        ("scatter", scatter, {"x": x, "y": y, "full": overlay_spec.get("full", False)})
                    )
                else:
                    x = np.asarray(overlay_spec["x"], dtype=float)
                    y = np.asarray(overlay_spec["y"], dtype=float)
                    line, = ax.plot(
                        [],
                        [],
                        color=color,
                        linewidth=overlay_spec.get("linewidth", 2.6),
                        linestyle=overlay_spec.get("linestyle", "-"),
                        alpha=overlay_spec.get("alpha", 0.95),
                        label=overlay_spec.get("label"),
                    )
                    overlay_artists.append(
                        ("line", line, {"x": x, "y": y, "full": overlay_spec.get("full", False)})
                    )

            if spec.get("threshold") is not None:
                threshold_value, threshold_label = spec["threshold"]
                ax.axhline(
                    y=float(threshold_value),
                    color="#ff5a5f",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.85,
                )
                x_left, x_right = ax.get_xlim()
                ax.text(
                    x_left + 0.02 * (x_right - x_left),
                    float(threshold_value) + 0.01,
                    str(threshold_label),
                    color="#ff8f8f",
                    fontsize=9,
                    ha="left",
                    va="bottom",
                )

            if any(getattr(artist, "get_label", lambda: None)() for _, artist, _ in artists + overlay_artists):
                ax.legend(loc="upper right", fontsize=8, framealpha=0.2)

            def _make_update(
                main_artists: list[tuple[str, Any, dict[str, Any]]],
                extra_artists: list[tuple[str, Any, dict[str, Any]]],
            ) -> Callable[[float], None]:
                def _update(progress: float) -> None:
                    for kind, artist, payload in main_artists:
                        if kind == "bar":
                            values = payload["values"]
                            for bar, value in zip(artist, values):
                                bar.set_height(progress * float(value))
                        elif kind == "scatter":
                            count = max(1, int(progress * len(payload["x"])))
                            points = np.column_stack([payload["x"][:count], payload["y"][:count]])
                            artist.set_offsets(points)
                        else:
                            count = max(1, int(progress * len(payload["x"])))
                            artist.set_data(payload["x"][:count], payload["y"][:count])

                    for kind, artist, payload in extra_artists:
                        if payload.get("full", False):
                            x = payload["x"]
                            y = payload["y"]
                            if kind == "scatter":
                                artist.set_offsets(np.column_stack([x, y]))
                            else:
                                artist.set_data(x, y)
                            continue

                        count = max(1, int(progress * len(payload["x"])))
                        if kind == "scatter":
                            points = np.column_stack([payload["x"][:count], payload["y"][:count]])
                            artist.set_offsets(points)
                        else:
                            artist.set_data(payload["x"][:count], payload["y"][:count])

                return _update

            plot_updates.append(_make_update(artists, overlay_artists))

        return fig, plot_updates

    def create_metrics_clip(
        self,
        plots: list[dict[str, Any]],
        kpi_overlay: dict[str, str],
        title_text: str,
        duration_sec: float = 10.0,
        fps: int = VIDEO_FPS,
    ) -> Path:
        """Generate the animated metrics clip."""
        output_path = self._metrics_path()
        frame_count = max(1, int(round(duration_sec * fps)))

        fig, plot_updates = self._build_metrics_figure(plots, kpi_overlay, title_text)
        intro_frames = max(1, int(0.18 * frame_count))

        def _update(frame_idx: int) -> list[Any]:
            if frame_idx <= intro_frames:
                progress = 0.0
            else:
                progress = _ease_in_out(
                    (frame_idx - intro_frames) / max(frame_count - intro_frames - 1, 1)
                )
            for update_fn in plot_updates:
                update_fn(progress)
            return []

        metrics_animation = animation.FuncAnimation(
            fig,
            _update,
            frames=frame_count,
            interval=1000 / fps,
            blit=False,
            repeat=False,
        )

        with _StreamingVideoWriter(output_path, VIDEO_WIDTH, VIDEO_HEIGHT, fps) as writer:
            for frame_idx in metrics_animation.new_frame_seq():
                metrics_animation._draw_next_frame(frame_idx, blit=False)
                writer.write(_figure_to_frame(fig))

        plt.close(fig)
        return output_path

    def record_simulation(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controller_fn: Callable[[mujoco.MjModel, mujoco.MjData, int], bool | None],
        camera_name: str = "fixed_top",
        playback_speed: float = 0.3,
        width: int = VIDEO_WIDTH,
        height: int = VIDEO_HEIGHT,
        fps: int = VIDEO_FPS,
        trace_ee: bool = True,
        duration_sec: float | None = None,
        *,
        camera_schedule: Callable[[float, float], Mapping[str, Any]] | None = None,
        trace_site_name: str | None = None,
        trace_project_z: float | None = None,
        planned_trace_points: np.ndarray | Sequence[Sequence[float]] | None = None,
        planned_trace_radius: float = 0.003,
        actual_trace_radius: float = 0.004,
        planned_trace_color: tuple[float, float, float, float] = (0.22, 0.92, 0.35, 0.90),
        actual_trace_color: tuple[float, float, float, float] = (0.22, 0.52, 0.98, 0.92),
        overlay_fn: Callable[[mujoco.MjvScene, mujoco.MjModel, mujoco.MjData, int], None] | None = None,
        status_text_fn: Callable[[mujoco.MjModel, mujoco.MjData, int, float], Mapping[str, Any] | Sequence[str] | str | None] | None = None,
        start_hold_sec: float = 1.0,
        end_hold_sec: float = 2.0,
    ) -> Path:
        """Record the MuJoCo simulation clip."""
        if playback_speed <= 0.0:
            raise ValueError("playback_speed must be positive.")

        output_path = self._simulation_path()
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera = _camera_from_name(model, camera_name)

        option = mujoco.MjvOption()
        option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

        sim_dt = float(model.opt.timestep)
        steps_per_frame = max(1, int(round(playback_speed / (fps * sim_dt))))
        max_steps = None if duration_sec is None else max(1, int(round(duration_sec / sim_dt)))

        trace_site_id = _resolve_trace_site_id(model, trace_site_name) if trace_ee else None
        planned_trace = None
        if planned_trace_points is not None:
            planned_trace = np.asarray(planned_trace_points, dtype=float)

        actual_trace: list[np.ndarray] = []
        self.last_simulation_trace = None

        def _render_current_frame(step_idx: int, current_time: float) -> np.ndarray:
            progress = 0.0 if duration_sec is None else current_time / max(duration_sec, sim_dt)
            if camera_schedule is not None:
                _apply_camera_state(camera, camera_schedule(current_time, float(np.clip(progress, 0.0, 1.0))))

            renderer.update_scene(data, camera=camera, scene_option=option)

            if planned_trace is not None and planned_trace.shape[0] > 1:
                add_polyline_to_scene(
                    renderer.scene,
                    planned_trace,
                    radius=planned_trace_radius,
                    rgba=planned_trace_color,
                    stride=1,
                    max_segments=280,
                )

            if actual_trace:
                add_polyline_to_scene(
                    renderer.scene,
                    np.asarray(actual_trace),
                    radius=actual_trace_radius,
                    rgba=actual_trace_color,
                    stride=max(1, len(actual_trace) // 180),
                    max_segments=240,
                )
                add_marker_to_scene(
                    renderer.scene,
                    np.asarray(actual_trace[-1], dtype=float),
                    size=actual_trace_radius * 1.7,
                    rgba=actual_trace_color,
                )

            if overlay_fn is not None:
                overlay_fn(renderer.scene, model, data, step_idx)

            frame = renderer.render().copy()
            if status_text_fn is not None:
                overlay_lines = _normalize_overlay_lines(
                    status_text_fn(model, data, step_idx, current_time)
                )
                frame = _draw_text_overlay(frame, overlay_lines)
            return frame

        with _StreamingVideoWriter(output_path, width, height, fps) as writer:
            mujoco.mj_forward(model, data)
            initial_time = 0.0
            if trace_site_id is not None:
                trace_point = data.site_xpos[trace_site_id].copy()
                if trace_project_z is not None:
                    trace_point[2] = trace_project_z
                actual_trace.append(trace_point)
            first_frame = _render_current_frame(0, initial_time)
            for _ in range(int(round(start_hold_sec * fps))):
                writer.write(first_frame)

            step = 0
            while True:
                result = controller_fn(model, data, step)
                mujoco.mj_step(model, data)
                step += 1

                if trace_site_id is not None:
                    trace_point = data.site_xpos[trace_site_id].copy()
                    if trace_project_z is not None:
                        trace_point[2] = trace_project_z
                    actual_trace.append(trace_point)

                current_time = step * sim_dt
                reached_duration = max_steps is not None and step >= max_steps
                completed = result is False or reached_duration

                if step % steps_per_frame == 0 or completed:
                    writer.write(_render_current_frame(step, current_time))

                if completed:
                    break

            end_frame = _render_current_frame(step, step * sim_dt)
            for _ in range(int(round(end_hold_sec * fps))):
                writer.write(end_frame)

        renderer.close()
        if actual_trace:
            self.last_simulation_trace = np.asarray(actual_trace, dtype=float)
        return output_path

    def compose_final_video(
        self,
        metrics_clip: Path,
        simulation_clip: Path,
        title_card_text: str | None = None,
        crossfade_sec: float = 0.5,
    ) -> Path:
        """Compose title card, metrics, simulation, and end card into one demo."""
        output_path = self._demo_path()
        title_card_text = title_card_text or self.lab_name

        temp_dir = Path(tempfile.mkdtemp(prefix=f"{self.file_prefix}_video_", dir=self.output_dir))
        try:
            title_clip = temp_dir / f"{self.file_prefix}_title.mp4"
            end_clip = temp_dir / f"{self.file_prefix}_end.mp4"

            title_frame = _make_card_frame(
                title_card_text,
                "MuJoCo Robotics Lab",
                footer="Metrics presentation and simulation playback",
            )
            end_frame = _make_card_frame(
                "End of Demo",
                self.lab_name,
                footer="Generated with the shared robotics video pipeline",
            )

            for clip_path, frame in ((title_clip, title_frame), (end_clip, end_frame)):
                with _StreamingVideoWriter(clip_path, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS) as writer:
                    for _ in range(int(round(CARD_DURATION_SEC * VIDEO_FPS))):
                        writer.write(frame)

            title_duration = CARD_DURATION_SEC
            metrics_duration = imageio_ffmpeg.count_frames_and_secs(str(metrics_clip))[1]
            simulation_duration = imageio_ffmpeg.count_frames_and_secs(str(simulation_clip))[1]
            end_duration = CARD_DURATION_SEC

            offset_1 = max(0.0, title_duration - crossfade_sec)
            combined_1 = title_duration + metrics_duration - crossfade_sec
            offset_2 = max(0.0, combined_1 - crossfade_sec)
            combined_2 = combined_1 + simulation_duration - crossfade_sec
            offset_3 = max(0.0, combined_2 - crossfade_sec)

            filter_graph = (
                f"[0:v][1:v]xfade=transition=fade:duration={crossfade_sec}:offset={offset_1:.3f}[v01];"
                f"[v01][2:v]xfade=transition=fade:duration={crossfade_sec}:offset={offset_2:.3f}[v012];"
                f"[v012][3:v]xfade=transition=fade:duration={crossfade_sec}:offset={offset_3:.3f}[vout]"
            )

            command = [
                FFMPEG_EXE,
                "-y",
                "-i",
                str(title_clip),
                "-i",
                str(metrics_clip),
                "-i",
                str(simulation_clip),
                "-i",
                str(end_clip),
                "-filter_complex",
                filter_graph,
                "-map",
                "[vout]",
                "-an",
                "-c:v",
                VIDEO_CODEC,
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
                str(output_path),
            ]

            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                stderr_tail = result.stderr[-1200:]
                raise RuntimeError(f"ffmpeg composition failed:\n{stderr_tail}")

            return output_path
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def generate_demo_video(
    lab_name: str,
    output_dir: Path,
    plots: list[dict[str, Any]],
    kpi_overlay: dict[str, str],
    metrics_title: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller_fn: Callable[[mujoco.MjModel, mujoco.MjData, int], bool | None],
    *,
    title_card_text: str | None = None,
    metrics_duration_sec: float = 10.0,
    metrics_fps: int = VIDEO_FPS,
    crossfade_sec: float = 0.5,
    camera_name: str = "fixed_top",
    playback_speed: float = 0.3,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    fps: int = VIDEO_FPS,
    trace_ee: bool = True,
    duration_sec: float | None = None,
    camera_schedule: Callable[[float, float], Mapping[str, Any]] | None = None,
    trace_site_name: str | None = None,
    trace_project_z: float | None = None,
    planned_trace_points: np.ndarray | Sequence[Sequence[float]] | None = None,
    overlay_fn: Callable[[mujoco.MjvScene, mujoco.MjModel, mujoco.MjData, int], None] | None = None,
    status_text_fn: Callable[[mujoco.MjModel, mujoco.MjData, int, float], Mapping[str, Any] | Sequence[str] | str | None] | None = None,
) -> Path:
    """Convenience wrapper that runs all three video phases."""
    producer = LabVideoProducer(lab_name=lab_name, output_dir=output_dir)
    metrics_clip = producer.create_metrics_clip(
        plots=plots,
        kpi_overlay=kpi_overlay,
        title_text=metrics_title,
        duration_sec=metrics_duration_sec,
        fps=metrics_fps,
    )
    simulation_clip = producer.record_simulation(
        model=model,
        data=data,
        controller_fn=controller_fn,
        camera_name=camera_name,
        playback_speed=playback_speed,
        width=width,
        height=height,
        fps=fps,
        trace_ee=trace_ee,
        duration_sec=duration_sec,
        camera_schedule=camera_schedule,
        trace_site_name=trace_site_name,
        trace_project_z=trace_project_z,
        planned_trace_points=planned_trace_points,
        overlay_fn=overlay_fn,
        status_text_fn=status_text_fn,
    )
    return producer.compose_final_video(
        metrics_clip=metrics_clip,
        simulation_clip=simulation_clip,
        title_card_text=title_card_text,
        crossfade_sec=crossfade_sec,
    )


__all__ = [
    "LabVideoProducer",
    "add_line_to_scene",
    "add_marker_to_scene",
    "add_polyline_to_scene",
    "generate_demo_video",
]
