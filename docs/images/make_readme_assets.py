"""Build every README asset: the downscaled stills and the two animated GIFs.

All sources are benchmark artefacts of one Spider 2.0-Lite task (``local008``)
that live under ``bench_results/`` in the benchmark checkout and stay out of
git, so the regeneration commands below need that checkout; the sources are
read, copied and resampled, never moved or modified:

* ``stills`` -- the LinkedIn renders in ``bench_results/linkedin_images/``,
  LANCZOS-downscaled to ~1200 px wide.
* ``graph-gif`` -- ``pipeline_walk.gif`` from the nine linker stages in
  ``bench_results/link_viz/stage0..8.png`` (regenerate the stages with
  ``uv run python bench_results/link_viz/make_link_animation.py``).
* ``tree-capture`` + ``tree-gif`` -- ``search_replay.gif``, captured by
  stepping the player of ``bench_results/replay_local008_multi20.html`` in
  headless Chrome (one invocation per step), then trimmed and quantized
  (regenerate the page with ``uv run schemagraph viz-search
  bench_results/spider2_exec_local008_multi20_candidates.jsonl``).

Run from the repo root (Pillow is undeclared: ``uv sync --all-extras`` pulls
it in through the abmcts-m chain; the capture needs a headless Chrome):

    uv run python docs/images/make_readme_assets.py all --bench bench_results
    uv run python docs/images/make_readme_assets.py stills
    uv run python docs/images/make_readme_assets.py graph-gif
    uv run python docs/images/make_readme_assets.py tree-capture
    uv run python docs/images/make_readme_assets.py tree-gif --frames DIR

Assets land in ``--out`` (default ``docs/images/``), captured frames in
``--frames`` (default ``$TMPDIR/schemagraph_readme_frames``, outside the repo).
Both GIFs must stay under :data:`GIF_MAX_BYTES`; if one grows past it, apply
the size ladder in order: trim pad 8 -> 4, capture 820x560, graph width
1200 -> 1000, palette 256 -> 128.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import subprocess
import tempfile
from pathlib import Path

try:
    from PIL import Image, ImageChops
except ImportError:  # pragma: no cover - environment guard for a fresh clone
    raise SystemExit(
        "Pillow is required and undeclared: run `uv sync --all-extras` (it arrives via "
        "abmcts-m -> PyMC -> arviz -> matplotlib) or `uv pip install pillow`, then re-run."
    ) from None

REPO_ROOT = Path(__file__).resolve().parents[2]

# Output still -> (source under --bench, target width in px; height keeps the aspect ratio).
STILL_SOURCES: dict[str, tuple[str, int]] = {
    "cover.png": ("linkedin_images/cover_1920x1080.png", 1200),
    "link_seeds.png": ("linkedin_images/1_graph_seeds.png", 1200),
    "link_pagerank.png": ("linkedin_images/2_graph_pagerank.png", 1200),
    "link_join_paths.png": ("linkedin_images/3_graph_join_paths.png", 1200),
    "link_tight_cut.png": ("linkedin_images/4_graph_tight_cut.png", 1200),
    "search_tree.png": ("linkedin_images/7_tree_five_models_20_nodes.png", 1200),
    "search_tree_details.png": ("linkedin_images/8_tree_five_models_with_details.png", 1200),
}

# The linker walk: nine stages, one GIF frame each (link_viz/make_link_animation.py).
GRAPH_FRAMES = tuple(f"link_viz/stage{i}.png" for i in range(9))
GRAPH_SIZE = (1200, 768)
GRAPH_DURATIONS_MS = [1800, 1600, 1600, 1600, 1400, 1600, 1400, 1600, 3200]

# The search replay: one viz-search page, stepped #forward once per frame.
REPLAY_SOURCE = "replay_local008_multi20.html"
REPLAY_REGEN = (
    "uv run schemagraph viz-search bench_results/spider2_exec_local008_multi20_candidates.jsonl"
)
LINK_VIZ_REGEN = "uv run python bench_results/link_viz/make_link_animation.py"
COVER_REGEN = "uv run python bench_results/linkedin_images/make_cover.py"

CAPTURE_WIDTH = 900  # headless-Chrome window and iframe size, in px
CAPTURE_HEIGHT = 600
CLICK_MS = 1500  # virtual-time between two #forward clicks
LOAD_MS = 3000  # virtual-time headroom for the page to load before the first click
SETTLE_MS = 3000  # virtual-time headroom after the last click, before the screenshot
CHROME_TIMEOUT_S = 180  # wall-clock cap per headless-Chrome invocation
CHROME_ATTEMPTS = 2  # screenshot retries per frame
TRIM_PAD = 8  # px of background kept around the frames' union bounding box
PALETTE_COLORS = 256
PALETTE_TILES = 4  # frames montaged to sample the shared GIF palette from
# (first, middle, last) frame durations of the tree GIF, in ms.
TREE_STEP_DURATIONS_MS = (1600, 600, 2600)
GIF_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_FRAMES = Path(tempfile.gettempdir()) / "schemagraph_readme_frames"

# Driver page: iframes the replay page, kills its transitions and chrome (header, facts,
# legend, side panel) so the 900x600 viewport is controls + caption + tree, then clicks
# #forward `steps` times on the virtual-time clock -- one screenshot per step count.
_DRIVER = string.Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<style>
html, body { margin: 0; padding: 0; background: #ffffff; }
iframe { display: block; border: 0; width: ${width}px; height: ${height}px; }
</style></head>
<body>
<iframe id="frame" src="${replay}"></iframe>
<script>
"use strict";
const steps = parseInt(new URLSearchParams(location.search).get("steps") || "0", 10);
const CLICK_MS = ${click_ms};
const OVERRIDE = "*, *::before, *::after { transition: none !important; "
  + "animation: none !important; }\\n"
  + "h1, .facts, .legend, .panel { display: none !important; }";
const frame = document.getElementById("frame");
frame.addEventListener("load", () => {
  const doc = frame.contentDocument;
  const style = doc.createElement("style");
  style.textContent = OVERRIDE;
  doc.head.appendChild(style);
  let done = 0;
  const tick = () => {
    if (done >= steps) { return; }
    const button = doc.getElementById("forward");
    if (!button) {
      document.body.textContent = "DRIVER ERROR: no #forward in the replay page";
      return;
    }
    button.click();
    done += 1;
    setTimeout(tick, CLICK_MS);
  };
  setTimeout(tick, CLICK_MS);
});
</script>
</body></html>
""")


def _require(path: Path, regen: str) -> Path:
    """Return `path`, or fail naming the missing input and how to regenerate it."""
    if not path.is_file():
        raise SystemExit(
            f"missing input: {path}\n"
            f"  regenerate it with: {regen}\n"
            "  (sources live under bench_results/ in the benchmark checkout, out of git;"
            " this script never moves them)"
        )
    return path


def _load(path: Path) -> Image.Image:
    """Open one source image as RGB."""
    with Image.open(path) as im:
        return im.convert("RGB")


def _still_regen(rel: str) -> str:
    """Pick the regeneration command that produced one still's source."""
    if "cover" in rel:
        return COVER_REGEN
    if rel.startswith(("linkedin_images/7", "linkedin_images/8")):
        return REPLAY_REGEN
    return LINK_VIZ_REGEN


def make_stills(bench: Path, out: Path) -> None:
    """Downscale the seven README stills from `bench` into `out`."""
    out.mkdir(parents=True, exist_ok=True)
    for name, (rel, width) in STILL_SOURCES.items():
        src = _require(bench / rel, _still_regen(rel))
        im = _load(src)
        height = round(im.height * width / im.width)
        im.resize((width, height), Image.Resampling.LANCZOS).save(
            out / name, optimize=True
        )
        print(f"  {name}: {width}x{height} from {src.name}")


def _global_palette(frames: list[Image.Image]) -> Image.Image:
    """Quantize a 2x2 montage of `PALETTE_TILES` evenly chosen frames to one palette."""
    last = len(frames) - 1
    picks = [frames[round(i * last / (PALETTE_TILES - 1))] for i in range(PALETTE_TILES)]
    w, h = picks[0].size
    montage = Image.new("RGB", (w * 2, h * 2))
    for i, im in enumerate(picks):
        montage.paste(im, ((i % 2) * w, (i // 2) * h))
    return montage.quantize(PALETTE_COLORS, method=Image.Quantize.MEDIANCUT)


def _save_gif(frames: list[Image.Image], durations: list[int], out: Path, ladder: str) -> None:
    """Save one looping, optimized GIF and fail past `GIF_MAX_BYTES`, naming the ladder."""
    frames[0].save(
        out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True
    )
    size = out.stat().st_size
    print(f"  {out.name}: {len(frames)} frames, {size / 1e6:.2f} MB")
    if size > GIF_MAX_BYTES:
        raise SystemExit(
            f"{out} is {size / 1e6:.2f} MB, over the {GIF_MAX_BYTES / 1e6:.0f} MB budget; "
            f"apply the size ladder in order: {ladder}"
        )


def make_graph_gif(bench: Path, out: Path) -> None:
    """Assemble `pipeline_walk.gif` from the nine linker stage renders in `bench`."""
    frames = [
        _load(_require(bench / rel, LINK_VIZ_REGEN)).resize(GRAPH_SIZE, Image.Resampling.LANCZOS)
        for rel in GRAPH_FRAMES
    ]
    palette = _global_palette(frames)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out.mkdir(parents=True, exist_ok=True)
    _save_gif(quantized, GRAPH_DURATIONS_MS, out / "pipeline_walk.gif",
              "graph width 1200 -> 1000, palette 256 -> 128")


def _event_count(replay: Path) -> int:
    """Read the replay page's embedded event JSON and return the number of steps."""
    src = replay.read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/json" id="search-events">(.*?)</script>', src, re.S
    )
    if match is None:
        raise SystemExit(f"{replay}: no #search-events block; regenerate with {REPLAY_REGEN}")
    # The block is plain JSON with backslash-u escapes (the renderer escapes <, > and &
    # that way too), so json.loads decodes it as is; the non-greedy regex is safe because
    # a literal "</script>" cannot occur inside the block.
    return len(json.loads(match.group(1))["events"])


def _capture_frame(driver: Path, out_png: Path, steps: int, chrome: str, profile: Path) -> None:
    """Screenshot the driver page once, after `steps` virtual-time #forward clicks."""
    budget = LOAD_MS + steps * CLICK_MS + SETTLE_MS
    cmd = [
        chrome, "--headless", "--disable-gpu", "--no-sandbox",
        "--allow-file-access-from-files", "--force-prefers-color-scheme=light",
        "--hide-scrollbars", f"--window-size={CAPTURE_WIDTH},{CAPTURE_HEIGHT}",
        f"--user-data-dir={profile}", "--no-first-run",
        f"--virtual-time-budget={budget}", f"--screenshot={out_png}",
        f"{driver.as_uri()}?steps={steps}",
    ]
    for attempt in range(1, CHROME_ATTEMPTS + 1):
        out_png.unlink(missing_ok=True)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=CHROME_TIMEOUT_S
            )
        except subprocess.TimeoutExpired:
            raise SystemExit(
                f"headless Chrome timed out after {CHROME_TIMEOUT_S}s on frame {steps}"
            ) from None
        if proc.returncode == 0 and out_png.is_file() and out_png.stat().st_size > 0:
            return
        if attempt == CHROME_ATTEMPTS:
            err = (proc.stderr or proc.stdout or "").strip()[-500:]
            raise SystemExit(
                f"headless Chrome failed on frame {steps} ({chrome}, {CHROME_ATTEMPTS} "
                f"attempts):\n{err}"
            )


def capture_tree(bench: Path, frames: Path, chrome: str) -> int:
    """Capture one frame per replay step (0..events) into `frames`; return the count."""
    replay = _require(bench / REPLAY_SOURCE, REPLAY_REGEN)
    events = _event_count(replay)
    frames.mkdir(parents=True, exist_ok=True)
    for old in frames.glob("frame_*.png"):
        old.unlink()
    driver = frames / "driver.html"
    try:
        html = _DRIVER.substitute(
            width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT,
            replay=replay.resolve().as_uri(), click_ms=CLICK_MS,
        )
    except ValueError as exc:  # a stray $ in the template or a substituted path
        raise SystemExit(f"driver template substitution failed: {exc}") from None
    driver.write_text(html, encoding="utf-8")
    profile = frames / "chrome-profile"
    for k in range(events + 1):
        _capture_frame(driver, frames / f"frame_{k:03d}.png", k, chrome, profile)
        print(f"  captured {k + 1}/{events + 1}", end="\r", flush=True)
    print()
    return events + 1


def make_tree_gif(frames: Path, out: Path) -> None:
    """Trim the captured frames to their union bbox and assemble `search_replay.gif`."""
    paths = sorted(frames.glob("frame_*.png"))
    if not paths:
        raise SystemExit(f"no frame_*.png under {frames}; run tree-capture first")
    if len(paths) < 3:
        raise SystemExit(f"need at least 3 frames under {frames}, found {len(paths)}")
    images = [_load(p) for p in paths]
    if all(im.tobytes() == images[0].tobytes() for im in images[1:]):
        raise SystemExit(
            f"every frame under {frames} is pixel-identical; the capture never stepped"
            " (check the driver's #forward clicks)"
        )
    background = Image.new("RGB", images[0].size, images[0].getpixel((0, 0)))
    box: tuple[int, int, int, int] | None = None
    for im in images:
        bbox = ImageChops.difference(im, background).getbbox()
        if bbox is None:
            continue
        box = bbox if box is None else (
            min(box[0], bbox[0]), min(box[1], bbox[1]),
            max(box[2], bbox[2]), max(box[3], bbox[3]),
        )
    if box is None:
        raise SystemExit(f"every frame under {frames} is pure background; the capture went wrong")
    width, height = images[0].size
    box = (
        max(box[0] - TRIM_PAD, 0), max(box[1] - TRIM_PAD, 0),
        min(box[2] + TRIM_PAD, width), min(box[3] + TRIM_PAD, height),
    )
    cropped = [im.crop(box) for im in images]
    palette = _global_palette(cropped)
    quantized = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in cropped]
    first, middle, last = TREE_STEP_DURATIONS_MS
    durations = [first] + [middle] * (len(quantized) - 2) + [last]
    out.mkdir(parents=True, exist_ok=True)
    print(f"  search_replay.gif: {len(quantized)} frames, trim {box[2] - box[0]}"
          f"x{box[3] - box[1]} at {box[:2]}")
    _save_gif(quantized, durations, out / "search_replay.gif",
              "trim pad 8 -> 4, capture 820x560, palette 256 -> 128")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line: one subcommand plus the shared path options."""
    parser = argparse.ArgumentParser(
        prog="make_readme_assets.py",
        description="Build the README's docs/images assets from bench_results sources"
                    " (benchmark checkout, out of git).",
    )
    # Shared options live on the subparsers, so flags follow the subcommand:
    # `make_readme_assets.py all --bench DIR`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bench", type=Path, default=REPO_ROOT / "bench_results",
                        help="bench_results directory of the benchmark checkout holding"
                             " the sources")
    common.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "images",
                        help="asset directory to write (default: docs/images)")
    common.add_argument("--chrome", default=os.environ.get("CHROME", "google-chrome"),
                        help="headless browser binary for tree-capture (default: $CHROME "
                             "or google-chrome)")
    common.add_argument("--frames", type=Path, default=DEFAULT_FRAMES,
                        help="frame directory for tree-capture / tree-gif (outside the repo)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in (
        ("stills", "downscale the seven stills"),
        ("graph-gif", "assemble pipeline_walk.gif from the nine stage renders"),
        ("tree-capture", "screenshot every replay step with headless Chrome (~42 runs, ~2 min)"),
        ("tree-gif", "assemble search_replay.gif from captured frames"),
        ("all", "stills, graph-gif, tree-capture, tree-gif in order"),
    ):
        sub.add_parser(name, parents=[common], help=help_text)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the chosen subcommand against `--bench`, writing assets to `--out`."""
    args = _parse_args(argv)
    bench: Path = args.bench
    if not bench.is_dir():
        raise SystemExit(f"--bench {bench} is not a directory (pass the bench_results path)")
    print(f"{args.cmd}: bench={bench} out={args.out}")
    if args.cmd in ("stills", "all"):
        make_stills(bench, args.out)
    if args.cmd in ("graph-gif", "all"):
        make_graph_gif(bench, args.out)
    if args.cmd in ("tree-capture", "all"):
        capture_tree(bench, args.frames, args.chrome)
    if args.cmd in ("tree-gif", "all"):
        make_tree_gif(args.frames, args.out)


if __name__ == "__main__":
    main()
