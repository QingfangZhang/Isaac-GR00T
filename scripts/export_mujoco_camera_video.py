#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterator
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/mujoco_mesa_shader_cache")
Path(os.environ["MESA_SHADER_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import mujoco  # noqa: E402


DEFAULT_REPLAY_DT = 0.0025
INDEX_TAG_RE = re.compile(r"\[(?P<tag>qpos|qvel)(?P<index>\d+)\]")


@dataclass
class RecordingEntry:
    recording_dir: Path
    csv_path: Path


@dataclass
class CsvMapping:
    scene_col: int = -1
    control_time_col: int = -1
    mujoco_time_col: int = -1
    qpos_cols: list[tuple[int, int]] | None = None


def clean_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser().absolute()


def path_is_under(path: Path, root: Path) -> bool:
    try:
        clean_path(path).relative_to(clean_path(root))
        return True
    except ValueError:
        return False


def is_recording_dir(path: Path) -> bool:
    return (path / "data.csv").is_file()


def find_recording_dirs(root: Path) -> list[RecordingEntry]:
    if is_recording_dir(root):
        return [RecordingEntry(root, root / "data.csv")]

    recordings = [
        RecordingEntry(csv_path.parent, csv_path)
        for csv_path in root.rglob("data.csv")
        if csv_path.is_file()
    ]
    recordings.sort(key=lambda entry: clean_path(entry.recording_dir).as_posix())
    return recordings


def parse_index_tag(header: str, tag: str) -> int:
    for match in INDEX_TAG_RE.finditer(header):
        if match.group("tag") == tag:
            return int(match.group("index"))
    return -1


def build_mapping(header: list[str]) -> CsvMapping:
    mapping = CsvMapping(qpos_cols=[])
    for col, name in enumerate(header):
        if name == "scene_path":
            mapping.scene_col = col
        elif name == "control_time_s":
            mapping.control_time_col = col
        elif name == "mujoco_time_s":
            mapping.mujoco_time_col = col

        qpos_idx = parse_index_tag(name, "qpos")
        if qpos_idx >= 0:
            mapping.qpos_cols.append((col, qpos_idx))

    if not mapping.qpos_cols:
        raise RuntimeError("CSV header has no qpos columns")
    return mapping


def parse_float(fields: list[str], col: int) -> float | None:
    if col < 0 or col >= len(fields) or not fields[col]:
        return None
    try:
        return float(fields[col])
    except ValueError:
        return None


def row_time(fields: list[str], mapping: CsvMapping, row_index: int) -> float:
    control_time = parse_float(fields, mapping.control_time_col)
    if control_time is not None:
        return control_time
    mujoco_time = parse_float(fields, mapping.mujoco_time_col)
    if mujoco_time is not None:
        return mujoco_time
    return row_index * DEFAULT_REPLAY_DT


def apply_frame(
    fields: list[str], mapping: CsvMapping, model: mujoco.MjModel, data: mujoco.MjData
) -> None:
    mujoco_time = parse_float(fields, mapping.mujoco_time_col)
    if mujoco_time is not None:
        data.time = mujoco_time

    assert mapping.qpos_cols is not None
    for col, qpos_idx in mapping.qpos_cols:
        if qpos_idx >= model.nq or col >= len(fields) or not fields[col]:
            continue
        try:
            data.qpos[qpos_idx] = float(fields[col])
        except ValueError:
            continue
    mujoco.mj_forward(model, data)


def iter_csv_rows(first_row: list[str], reader: csv.reader) -> Iterator[tuple[int, list[str]]]:
    yield 0, first_row
    yield from enumerate(reader, start=1)


def find_scene_in_recording_dir(recording_dir: Path) -> Path:
    snapshot_dir = recording_dir / "model_snapshot"
    if not snapshot_dir.is_dir():
        raise RuntimeError(f"recording directory has no model_snapshot directory: {recording_dir}")

    candidates = [
        path for path in snapshot_dir.rglob("*.xml") if path.is_file() and "scene" in path.name
    ]
    if not candidates:
        candidates = [path for path in snapshot_dir.rglob("*.xml") if path.is_file()]
    if not candidates:
        raise RuntimeError(f"no XML scene found under {snapshot_dir}")
    return sorted(candidates)[0]


def resolve_recording_scene_path(scene_from_csv: str, recording_dir: Path) -> Path:
    if scene_from_csv:
        csv_scene = Path(scene_from_csv)
        if csv_scene.exists() and path_is_under(csv_scene, recording_dir):
            return csv_scene

        parts = list(csv_scene.parts)
        if "model_snapshot" in parts:
            marker = parts.index("model_snapshot")
            candidate = recording_dir.joinpath(*parts[marker:])
            if candidate.exists():
                return candidate

    return find_scene_in_recording_dir(recording_dir)


def snapshot_root_from_scene(scene_path: Path) -> Path | None:
    root = Path(scene_path.root)
    for part in clean_path(scene_path).parts:
        if part == scene_path.root:
            continue
        root /= part
        if part == "model_snapshot":
            return root
    return None


def ensure_directory_link(target: Path, link: Path) -> None:
    if not target.is_dir():
        return
    if link.is_symlink():
        if clean_path(link.resolve()) == clean_path(target):
            return
        link.unlink()
    elif link.exists():
        return

    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def prepare_snapshot_assets(scene_path: Path, snapshot_root: Path, data_root: Path) -> None:
    scene_dir = scene_path.parent
    mujoco_root = snapshot_root / "mujoco"
    model_root = mujoco_root / "model"
    repo_model_root = data_root / "mujoco" / "model"

    for name in ("meshes", "textures", "objects"):
        ensure_directory_link(repo_model_root / "g1" / name, scene_dir / name)

    for name in ("task_assets", "adam_pro", "adam_pick", "robotwin_assets"):
        ensure_directory_link(repo_model_root / name, scene_dir / name)
        ensure_directory_link(repo_model_root / name, model_root / name)
        ensure_directory_link(repo_model_root / name, mujoco_root / name)


def prepare_replay_scene(scene_path: Path, data_root: Path) -> tuple[Path, Path | None]:
    snapshot_root = snapshot_root_from_scene(scene_path)
    if snapshot_root is None:
        return scene_path, None

    temp_dir = Path(tempfile.mkdtemp(prefix="mujoco_camera_export_"))
    temp_snapshot_root = temp_dir / "model_snapshot"
    snapshot_root_abs = clean_path(snapshot_root)

    for xml_path in snapshot_root_abs.rglob("*.xml"):
        rel = xml_path.relative_to(snapshot_root_abs)
        dst = temp_snapshot_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(xml_path, dst)

    scene_rel = clean_path(scene_path).relative_to(snapshot_root_abs)
    replay_scene = temp_snapshot_root / scene_rel
    prepare_snapshot_assets(replay_scene, temp_snapshot_root, data_root)
    return replay_scene, temp_dir


def camera_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for camera_id in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        if name:
            names.append(name)
    return names


def open_ffmpeg(
    output_path: Path, width: int, height: int, fps: float, crf: int
) -> subprocess.Popen:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def recording_output_path(
    recording: RecordingEntry, input_root: Path, output: Path, camera: str
) -> Path:
    if output.suffix.lower() == ".mp4":
        return output
    if is_recording_dir(input_root):
        stem = recording.recording_dir.name
    else:
        try:
            stem = recording.recording_dir.relative_to(input_root).as_posix().replace("/", "__")
        except ValueError:
            stem = recording.recording_dir.name
    return output / f"{stem}.{camera}.mp4"


def export_recording(
    recording: RecordingEntry,
    output_path: Path,
    camera: str,
    width: int,
    height: int,
    fps: float,
    max_duration: float | None,
    start_time: float,
    crf: int,
    overwrite: bool,
    data_root: Path,
) -> int:
    if output_path.exists() and not overwrite:
        print(f"[skip] exists: {output_path}")
        return 0

    with recording.csv_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"empty CSV: {recording.csv_path}")
        mapping = build_mapping(header)

        first_row = next(reader, None)
        if first_row is None:
            raise RuntimeError(f"CSV has header but no samples: {recording.csv_path}")

        scene_value = first_row[mapping.scene_col] if mapping.scene_col >= 0 else ""
        scene_path = resolve_recording_scene_path(scene_value, recording.recording_dir)
        replay_scene, temp_dir = prepare_replay_scene(scene_path, data_root)

        try:
            model = mujoco.MjModel.from_xml_path(str(replay_scene))
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if cam_id < 0:
                raise RuntimeError(
                    f"camera '{camera}' not found in {scene_path}; available: {camera_names(model)}"
                )

            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=height, width=width)
            ffmpeg = open_ffmpeg(output_path, width, height, fps, crf)
            assert ffmpeg.stdin is not None

            first_t = row_time(first_row, mapping, 0)
            next_render_time = start_time
            rendered = 0

            for row_index, fields in iter_csv_rows(first_row, reader):
                current_t = row_time(fields, mapping, row_index)
                rel_t = max(0.0, current_t - first_t)
                if rel_t + 1e-9 < next_render_time:
                    continue
                if max_duration is not None and rel_t > start_time + max_duration:
                    break

                apply_frame(fields, mapping, model, data)
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                ffmpeg.stdin.write(frame.tobytes())
                rendered += 1
                next_render_time = start_time + rendered / fps

            ffmpeg.stdin.close()
            return_code = ffmpeg.wait()
            renderer.close()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {output_path}")
            print(f"[ok] {output_path} frames={rendered} camera={camera}")
            return rendered
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)


def list_recording_cameras(recording: RecordingEntry, data_root: Path) -> None:
    with recording.csv_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        first_row = next(reader, None)
    if header is None or first_row is None:
        raise RuntimeError(f"empty CSV: {recording.csv_path}")
    mapping = build_mapping(header)
    scene_value = first_row[mapping.scene_col] if mapping.scene_col >= 0 else ""
    scene_path = resolve_recording_scene_path(scene_value, recording.recording_dir)
    replay_scene, temp_dir = prepare_replay_scene(scene_path, data_root)
    try:
        model = mujoco.MjModel.from_xml_path(str(replay_scene))
        print(f"{recording.recording_dir}:")
        for name in camera_names(model):
            print(f"  {name}")
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MuJoCo fixed-camera video from replay_mujoco_csv recordings."
    )
    parser.add_argument("input", type=Path, help="recording dir, data.csv, or parent directory")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="MuJoCo camera name; repeat to export multiple cameras",
    )
    parser.add_argument("--output", type=Path, default=Path("output/mujoco_camera_videos"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-duration", type=float, default=None, help="seconds per recording")
    parser.add_argument(
        "--start-time", type=float, default=0.0, help="seconds from recording start"
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--limit", type=int, default=None, help="maximum recordings to export")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-cameras", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(__file__).resolve().parents[1] / "data"
    input_path = args.input
    if input_path.is_file():
        recordings = [RecordingEntry(input_path.parent, input_path)]
        input_root = input_path.parent
    else:
        input_root = input_path
        recordings = find_recording_dirs(input_path)

    if args.limit is not None:
        recordings = recordings[: args.limit]
    if not recordings:
        raise RuntimeError(f"no recordings found: {input_path}")

    if args.list_cameras:
        for recording in recordings:
            list_recording_cameras(recording, data_root)
        return 0

    cameras = args.camera or ["head_camera"]
    tasks = [(recording, camera) for recording in recordings for camera in cameras]
    if args.output.suffix.lower() == ".mp4" and len(tasks) != 1:
        raise RuntimeError(
            "--output can be an .mp4 file only when exporting one recording and one camera"
        )

    total_frames = 0
    for recording, camera in tasks:
        output_path = recording_output_path(recording, input_root, args.output, camera)
        total_frames += export_recording(
            recording=recording,
            output_path=output_path,
            camera=camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            max_duration=args.max_duration,
            start_time=args.start_time,
            crf=args.crf,
            overwrite=args.overwrite,
            data_root=data_root,
        )
    print(f"[done] recordings={len(recordings)} cameras={len(cameras)} frames={total_frames}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
