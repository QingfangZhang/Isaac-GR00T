#!/usr/bin/env python3
"""Convert replay_mujoco_csv recordings into GR00T LeRobot v2 episodes.

Each output sample is keyed by a unique 50 Hz ``policy_seq``. The policy payload is
paired with the nearest preceding 400 Hz MuJoCo state whose robot joint positions
match ``policy_received_dof_pos``. That same MuJoCo state drives the ego-view video
frame and projected-gravity calculation.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


DEFAULT_REPLAY_DT = 0.0025
EXPECTED_FPS = 50.0
EXPECTED_TOKEN_SIZE = 64
EXPECTED_REFERENCE_MOTION_SIZE = 640
EXPECTED_STATE_SIZE = 43
EXPECTED_HAND_SIZE = 7
CHUNK_SIZE = 1000

QPOS_INDEX_RE = re.compile(r"\[qpos(?P<index>\d+)\]")

STATE_GROUPS = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "right_arm": (22, 29),
    "left_hand": (29, 36),
    "right_hand": (36, 43),
}

HAND_JOINT_NAMES = [
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
]


@dataclass(frozen=True)
class Recording:
    directory: Path
    csv_path: Path


@dataclass(frozen=True)
class CsvMapping:
    scene_col: int
    control_time_col: int
    mujoco_time_col: int
    policy_valid_col: int
    policy_seq_col: int
    policy_token_size_col: int
    policy_reference_motion_size_col: int
    qpos_cols: tuple[tuple[int, int], ...]
    qpos_names: tuple[str, ...]
    received_state_cols: tuple[int, ...]
    token_cols: tuple[int, ...]
    left_hand_cols: tuple[int, ...]
    right_hand_cols: tuple[int, ...]


@dataclass(frozen=True)
class StateSnapshot:
    row_index: int
    control_time: float
    mujoco_time: float
    qpos: np.ndarray


@dataclass(frozen=True)
class AlignedSample:
    policy_seq: int
    policy_row_index: int
    observation_row_index: int
    policy_control_time: float
    observation_control_time: float
    mujoco_time: float
    lag_rows: int
    lag_seconds: float
    state_match_rmse: float
    state_match_max_abs: float
    qpos: np.ndarray
    observation_state: np.ndarray
    projected_gravity: np.ndarray
    motion_token: np.ndarray
    left_hand_action: np.ndarray
    right_hand_action: np.ndarray


@dataclass(frozen=True)
class ScanResult:
    recording: Recording
    scene_value: str
    source_rows: int
    samples: tuple[AlignedSample, ...]
    state_names: tuple[str, ...]
    skipped_boundary_policy_frames: int


@dataclass(frozen=True)
class VideoInfo:
    codec: str
    pixel_format: str
    width: int
    height: int
    fps: float
    frames: int
    camera_fovy: float
    camera_position: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class EpisodeSummary:
    episode_index: int
    recording_name: str
    source_csv: str
    task_index: int
    task: str
    frames: int
    first_policy_seq: int
    last_policy_seq: int
    mean_lag_ms: float
    max_state_match_error: float
    skipped_boundary_policy_frames: int
    video: VideoInfo


def find_recordings(input_path: Path) -> list[Recording]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.name != "data.csv":
            raise ValueError(f"input file must be named data.csv: {input_path}")
        return [Recording(input_path.parent, input_path)]
    if (input_path / "data.csv").is_file():
        return [Recording(input_path, input_path / "data.csv")]

    recordings = [
        Recording(path.parent, path) for path in input_path.rglob("data.csv") if path.is_file()
    ]
    recordings.sort(key=lambda item: item.directory.as_posix())
    if not recordings:
        raise ValueError(f"no data.csv recordings found under {input_path}")
    return recordings


def _required_column(header: list[str], name: str) -> int:
    try:
        return header.index(name)
    except ValueError as exc:
        raise ValueError(f"required CSV column is missing: {name}") from exc


def _indexed_columns(header: list[str], prefix: str, expected_size: int | None) -> tuple[int, ...]:
    pattern = re.compile(rf"^{re.escape(prefix)}\[(?P<index>\d+)\]$")
    indexed: dict[int, int] = {}
    for column, name in enumerate(header):
        match = pattern.match(name)
        if match:
            indexed[int(match.group("index"))] = column

    if expected_size is None:
        expected_size = max(indexed, default=-1) + 1
    expected_indices = set(range(expected_size))
    missing = sorted(expected_indices.difference(indexed))
    if missing:
        raise ValueError(f"{prefix} is missing indices: {missing[:10]}")
    return tuple(indexed[index] for index in range(expected_size))


def _joint_name_from_qpos_header(name: str, qpos_index: int) -> str:
    descriptor = name.split(":", 1)[-1].split("[qpos", 1)[0]
    parts = descriptor.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return f"qpos_{qpos_index}"


def build_csv_mapping(header: list[str]) -> CsvMapping:
    qpos_by_index: dict[int, int] = {}
    qpos_name_by_index: dict[int, str] = {}
    for column, name in enumerate(header):
        match = QPOS_INDEX_RE.search(name)
        if match:
            qpos_index = int(match.group("index"))
            qpos_by_index[qpos_index] = column
            qpos_name_by_index[qpos_index] = _joint_name_from_qpos_header(name, qpos_index)

    if not qpos_by_index or min(qpos_by_index) != 0:
        raise ValueError("CSV has no contiguous qpos columns starting at qpos0")
    qpos_size = max(qpos_by_index) + 1
    missing_qpos = sorted(set(range(qpos_size)).difference(qpos_by_index))
    if missing_qpos:
        raise ValueError(f"qpos is missing indices: {missing_qpos[:10]}")
    if qpos_size < 50:
        raise ValueError(f"at least 50 qpos values are required, found {qpos_size}")

    return CsvMapping(
        scene_col=_required_column(header, "scene_path"),
        control_time_col=_required_column(header, "control_time_s"),
        mujoco_time_col=_required_column(header, "mujoco_time_s"),
        policy_valid_col=_required_column(header, "policy_valid"),
        policy_seq_col=_required_column(header, "policy_seq"),
        policy_token_size_col=_required_column(header, "policy_token_size"),
        policy_reference_motion_size_col=_required_column(header, "policy_reference_motion_size"),
        qpos_cols=tuple((qpos_by_index[index], index) for index in range(qpos_size)),
        qpos_names=tuple(qpos_name_by_index[index] for index in range(qpos_size)),
        received_state_cols=_indexed_columns(
            header, "policy_received_dof_pos", EXPECTED_STATE_SIZE
        ),
        token_cols=_indexed_columns(header, "token_state", None),
        left_hand_cols=_indexed_columns(header, "left_hand_q", EXPECTED_HAND_SIZE),
        right_hand_cols=_indexed_columns(header, "right_hand_q", EXPECTED_HAND_SIZE),
    )


def _parse_float(fields: list[str], column: int, name: str) -> float:
    try:
        value = float(fields[column])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"invalid {name}: {fields[column] if column < len(fields) else None}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}: {value}")
    return value


def _parse_int(fields: list[str], column: int, name: str) -> int:
    value = _parse_float(fields, column, name)
    rounded = round(value)
    if not math.isclose(value, rounded, abs_tol=1e-6):
        raise ValueError(f"{name} must be integral, got {value}")
    return int(rounded)


def _parse_vector(
    fields: list[str], columns: tuple[int, ...], name: str, dtype: np.dtype[Any]
) -> np.ndarray:
    values = np.asarray([_parse_float(fields, column, name) for column in columns], dtype=dtype)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return values


def _parse_qpos(fields: list[str], mapping: CsvMapping) -> np.ndarray:
    qpos = np.empty(len(mapping.qpos_cols), dtype=np.float64)
    for column, qpos_index in mapping.qpos_cols:
        qpos[qpos_index] = _parse_float(fields, column, f"qpos[{qpos_index}]")
    return qpos


def qpos_to_policy_state(qpos: np.ndarray) -> np.ndarray:
    """Reorder MuJoCo robot qpos into body-29, left-hand-7, right-hand-7 order."""
    if qpos.shape[0] < 50:
        raise ValueError(f"qpos needs at least 50 elements, got {qpos.shape[0]}")
    return np.concatenate([qpos[7:29], qpos[36:43], qpos[29:36], qpos[43:50]]).astype(np.float32)


def state_names_from_mapping(mapping: CsvMapping) -> tuple[str, ...]:
    indices = [*range(7, 29), *range(36, 43), *range(29, 36), *range(43, 50)]
    return tuple(mapping.qpos_names[index] for index in indices)


def compute_projected_gravity(base_quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate world gravity [0, 0, -1] into the robot body frame."""
    quat = np.asarray(base_quat_wxyz, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"base quaternion must have shape (4,), got {quat.shape}")
    norm = np.linalg.norm(quat)
    if not math.isfinite(float(norm)) or norm < 1e-12:
        raise ValueError(f"invalid base quaternion norm: {norm}")
    w, x, y, z = quat / norm
    rotation_body_to_world = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return (rotation_body_to_world.T @ gravity_world).astype(np.float32)


def find_best_observation_snapshot(
    snapshots: deque[StateSnapshot], received_state: np.ndarray
) -> tuple[StateSnapshot, float, float]:
    if not snapshots:
        raise ValueError("cannot align policy state without MuJoCo snapshots")

    best: tuple[float, int, StateSnapshot, float] | None = None
    for snapshot in snapshots:
        difference = qpos_to_policy_state(snapshot.qpos).astype(np.float64) - received_state
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        max_abs = float(np.max(np.abs(difference)))
        candidate = (rmse, -snapshot.row_index, snapshot, max_abs)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[2], best[0], best[3]


def scan_recording(
    recording: Recording,
    state_match_window_rows: int,
    max_state_match_error: float,
    allow_policy_seq_gaps: bool,
) -> ScanResult:
    if state_match_window_rows < 1:
        raise ValueError("state_match_window_rows must be at least 1")

    samples: list[AlignedSample] = []
    snapshots: deque[StateSnapshot] = deque(maxlen=state_match_window_rows + 1)
    scene_value = ""
    source_rows = 0
    last_seen_policy_seq: int | None = None
    skipped_boundary_policy_frames = 0

    with recording.csv_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"empty CSV: {recording.csv_path}")
        mapping = build_csv_mapping(header)

        for row_index, fields in enumerate(reader):
            source_rows += 1
            if len(fields) != len(header):
                raise ValueError(
                    f"{recording.csv_path}:{row_index + 2} has {len(fields)} fields, "
                    f"expected {len(header)}"
                )

            if row_index == 0:
                scene_value = fields[mapping.scene_col]

            control_time = _parse_float(fields, mapping.control_time_col, "control_time_s")
            mujoco_time = _parse_float(fields, mapping.mujoco_time_col, "mujoco_time_s")
            qpos = _parse_qpos(fields, mapping)
            snapshot = StateSnapshot(row_index, control_time, mujoco_time, qpos)
            snapshots.append(snapshot)

            if _parse_int(fields, mapping.policy_valid_col, "policy_valid") != 1:
                continue
            policy_seq = _parse_int(fields, mapping.policy_seq_col, "policy_seq")
            if policy_seq == last_seen_policy_seq:
                continue
            if last_seen_policy_seq is not None:
                if policy_seq < last_seen_policy_seq:
                    raise ValueError(
                        f"policy_seq decreased from {last_seen_policy_seq} to {policy_seq} "
                        f"at row {row_index}"
                    )
                if policy_seq != last_seen_policy_seq + 1 and not allow_policy_seq_gaps:
                    raise ValueError(
                        f"policy_seq gap from {last_seen_policy_seq} to {policy_seq} "
                        f"at row {row_index}"
                    )
            last_seen_policy_seq = policy_seq

            token_size = _parse_int(fields, mapping.policy_token_size_col, "policy_token_size")
            if token_size != EXPECTED_TOKEN_SIZE:
                raise ValueError(
                    f"policy_token_size must be {EXPECTED_TOKEN_SIZE}, got {token_size} "
                    f"at row {row_index}"
                )
            reference_size = _parse_int(
                fields,
                mapping.policy_reference_motion_size_col,
                "policy_reference_motion_size",
            )
            if reference_size != EXPECTED_REFERENCE_MOTION_SIZE:
                raise ValueError(
                    f"policy_reference_motion_size must be {EXPECTED_REFERENCE_MOTION_SIZE}, "
                    f"got {reference_size} at row {row_index}"
                )

            received_state = _parse_vector(
                fields, mapping.received_state_cols, "policy_received_dof_pos", np.float32
            )
            all_tokens = _parse_vector(fields, mapping.token_cols, "token_state", np.float32)
            if not np.allclose(all_tokens[token_size:], 0.0, atol=1e-7):
                raise ValueError(f"non-zero token padding at row {row_index}")
            left_hand = _parse_vector(fields, mapping.left_hand_cols, "left_hand_q", np.float32)
            right_hand = _parse_vector(fields, mapping.right_hand_cols, "right_hand_q", np.float32)

            observation, match_rmse, match_max_abs = find_best_observation_snapshot(
                snapshots, received_state
            )
            if match_max_abs > max_state_match_error:
                if not samples and row_index < state_match_window_rows:
                    skipped_boundary_policy_frames += 1
                    continue
                raise ValueError(
                    f"state alignment error {match_max_abs:.6f} rad exceeds "
                    f"{max_state_match_error:.6f} at policy_seq {policy_seq}"
                )

            lag_rows = row_index - observation.row_index
            samples.append(
                AlignedSample(
                    policy_seq=policy_seq,
                    policy_row_index=row_index,
                    observation_row_index=observation.row_index,
                    policy_control_time=control_time,
                    observation_control_time=observation.control_time,
                    mujoco_time=observation.mujoco_time,
                    lag_rows=lag_rows,
                    lag_seconds=control_time - observation.control_time,
                    state_match_rmse=match_rmse,
                    state_match_max_abs=match_max_abs,
                    qpos=observation.qpos.copy(),
                    observation_state=received_state,
                    projected_gravity=compute_projected_gravity(observation.qpos[3:7]),
                    motion_token=all_tokens[:token_size].copy(),
                    left_hand_action=left_hand,
                    right_hand_action=right_hand,
                )
            )

    if not samples:
        raise ValueError(f"recording has no valid policy samples: {recording.csv_path}")
    if len(samples) < 40:
        raise ValueError(
            f"recording has only {len(samples)} policy frames; UNITREE_G1_SONIC needs at least 40"
        )
    return ScanResult(
        recording=recording,
        scene_value=scene_value,
        source_rows=source_rows,
        samples=tuple(samples),
        state_names=state_names_from_mapping(mapping),
        skipped_boundary_policy_frames=skipped_boundary_policy_frames,
    )


def _load_replay_video_module():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_mujoco_camera_video

    return export_mujoco_camera_video


def _parse_rate(rate: str) -> float:
    numerator, separator, denominator = rate.partition("/")
    if not separator:
        return float(rate)
    denominator_value = float(denominator)
    if denominator_value == 0:
        return 0.0
    return float(numerator) / denominator_value


def probe_video(
    path: Path,
    expected_frames: int,
    camera_fovy: float,
    camera_position: tuple[float, float, float] | None = None,
) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    frame_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    frames = int(frame_value)
    if frames != expected_frames:
        raise ValueError(f"video has {frames} frames, expected {expected_frames}: {path}")
    return VideoInfo(
        codec=str(stream["codec_name"]),
        pixel_format=str(stream["pix_fmt"]),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=_parse_rate(str(stream["avg_frame_rate"])),
        frames=frames,
        camera_fovy=camera_fovy,
        camera_position=camera_position,
    )


def render_video(
    scan: ScanResult,
    output_path: Path,
    camera: str,
    width: int,
    height: int,
    fps: float,
    crf: int,
    data_root: Path,
    camera_fovy: float | None = None,
    camera_position: tuple[float, float, float] | None = None,
) -> VideoInfo:
    replay_video = _load_replay_video_module()
    mujoco = replay_video.mujoco
    scene_path = replay_video.resolve_recording_scene_path(
        scan.scene_value, scan.recording.directory
    )
    replay_scene, temp_scene_dir = replay_video.prepare_replay_scene(scene_path, data_root)
    renderer = None
    ffmpeg = None
    try:
        model = mujoco.MjModel.from_xml_path(str(replay_scene))
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if camera_id < 0:
            raise ValueError(
                f"camera '{camera}' not found in {scene_path}; "
                f"available: {replay_video.camera_names(model)}"
            )
        if camera_fovy is not None:
            model.cam_fovy[camera_id] = camera_fovy
        if camera_position is not None:
            model.cam_pos[camera_id] = camera_position
        if scan.samples[0].qpos.shape[0] < model.nq:
            raise ValueError(
                f"CSV qpos size {scan.samples[0].qpos.shape[0]} is smaller than model.nq {model.nq}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        ffmpeg = replay_video.open_ffmpeg(output_path, width, height, fps, crf)
        assert ffmpeg.stdin is not None

        for sample in scan.samples:
            data.time = sample.mujoco_time
            data.qpos[:] = sample.qpos[: model.nq]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = np.ascontiguousarray(renderer.render(), dtype=np.uint8)
            ffmpeg.stdin.write(frame.tobytes())

        ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {output_path}")
        ffmpeg = None
        actual_camera_position = tuple(float(value) for value in model.cam_pos[camera_id])
        return probe_video(
            output_path,
            len(scan.samples),
            float(model.cam_fovy[camera_id]),
            actual_camera_position,
        )
    finally:
        if ffmpeg is not None:
            if ffmpeg.stdin is not None and not ffmpeg.stdin.closed:
                ffmpeg.stdin.close()
            ffmpeg.terminate()
            ffmpeg.wait()
        if renderer is not None:
            renderer.close()
        if temp_scene_dir is not None:
            shutil.rmtree(temp_scene_dir, ignore_errors=True)


def _fixed_list_array(values: list[np.ndarray], size: int):
    import pyarrow as pa

    return pa.array([value.tolist() for value in values], type=pa.list_(pa.float32(), size))


def write_episode_parquet(
    path: Path,
    samples: tuple[AlignedSample, ...],
    episode_index: int,
    task_index: int,
    global_start_index: int,
    fps: float,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    length = len(samples)
    columns = {
        "observation.state": _fixed_list_array(
            [sample.observation_state for sample in samples], EXPECTED_STATE_SIZE
        ),
        "observation.projected_gravity": _fixed_list_array(
            [sample.projected_gravity for sample in samples], 3
        ),
        "action.motion_token": _fixed_list_array(
            [sample.motion_token for sample in samples], EXPECTED_TOKEN_SIZE
        ),
        "teleop.left_hand_joints": _fixed_list_array(
            [sample.left_hand_action for sample in samples], EXPECTED_HAND_SIZE
        ),
        "teleop.right_hand_joints": _fixed_list_array(
            [sample.right_hand_action for sample in samples], EXPECTED_HAND_SIZE
        ),
        "timestamp": pa.array(np.arange(length, dtype=np.float32) / fps, type=pa.float32()),
        "annotation.human.task_description": pa.array(
            np.full(length, task_index, dtype=np.int64), type=pa.int64()
        ),
        "task_index": pa.array(np.full(length, task_index, dtype=np.int64), type=pa.int64()),
        "frame_index": pa.array(np.arange(length, dtype=np.int64), type=pa.int64()),
        "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()),
        "index": pa.array(
            np.arange(global_start_index, global_start_index + length, dtype=np.int64),
            type=pa.int64(),
        ),
        "next.reward": pa.array(np.zeros(length, dtype=np.float32), type=pa.float32()),
        "next.done": pa.array(np.arange(length, dtype=np.int64) == length - 1, type=pa.bool_()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, compression="zstd")
    rows = pq.ParquetFile(path).metadata.num_rows
    if rows != length:
        raise ValueError(f"Parquet has {rows} rows, expected {length}: {path}")
    return rows


def _feature(dtype: str, shape: list[int], names: list[str] | None) -> dict[str, Any]:
    return {"dtype": dtype, "shape": shape, "names": names}


def build_info(
    summaries: list[EpisodeSummary],
    state_names: tuple[str, ...],
    fps: float,
    width: int,
    height: int,
    task_count: int,
) -> dict[str, Any]:
    total_frames = sum(summary.frames for summary in summaries)
    features: dict[str, Any] = {
        "observation.state": _feature("float32", [43], list(state_names)),
        "observation.projected_gravity": _feature(
            "float32", [3], ["gravity_x", "gravity_y", "gravity_z"]
        ),
        "action.motion_token": _feature(
            "float32", [64], [f"motion_token_{index}" for index in range(64)]
        ),
        "teleop.left_hand_joints": _feature("float32", [7], HAND_JOINT_NAMES),
        "teleop.right_hand_joints": _feature("float32", [7], HAND_JOINT_NAMES),
        "timestamp": _feature("float32", [1], None),
        "annotation.human.task_description": _feature("int64", [1], None),
        "task_index": _feature("int64", [1], None),
        "frame_index": _feature("int64", [1], None),
        "episode_index": _feature("int64", [1], None),
        "index": _feature("int64", [1], None),
        "next.reward": _feature("float32", [1], None),
        "next.done": _feature("bool", [1], None),
    }
    features["observation.images.ego_view"] = {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    return {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1_sonic",
        "total_episodes": len(summaries),
        "total_frames": total_frames,
        "total_tasks": task_count,
        "total_videos": len(summaries),
        "total_chunks": math.ceil(len(summaries) / CHUNK_SIZE),
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{len(summaries)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": features,
    }


def build_modality() -> dict[str, Any]:
    state = {
        key: {"start": start, "end": end, "original_key": "observation.state"}
        for key, (start, end) in STATE_GROUPS.items()
    }
    state["projected_gravity"] = {
        "start": 0,
        "end": 3,
        "original_key": "observation.projected_gravity",
    }
    return {
        "state": state,
        "action": {
            "motion_token": {
                "start": 0,
                "end": 64,
                "original_key": "action.motion_token",
            },
            "left_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": "teleop.left_hand_joints",
            },
            "right_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": "teleop.right_hand_joints",
            },
        },
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {
            "human.task_description": {"original_key": "annotation.human.task_description"}
        },
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def _read_episode_summary(path: Path) -> EpisodeSummary:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.setdefault("skipped_boundary_policy_frames", 0)
    camera_position = value["video"].get("camera_position")
    if camera_position is not None:
        value["video"]["camera_position"] = tuple(camera_position)
    value["video"] = VideoInfo(**value["video"])
    return EpisodeSummary(**value)


def _parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def _jsonl(values: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(value, ensure_ascii=True) + "\n" for value in values)


def write_alignment_audit(path: Path, samples: tuple[AlignedSample, ...]) -> None:
    rows = []
    for frame_index, sample in enumerate(samples):
        rows.append(
            {
                "frame_index": frame_index,
                "policy_seq": sample.policy_seq,
                "policy_row_index": sample.policy_row_index,
                "observation_row_index": sample.observation_row_index,
                "policy_control_time": sample.policy_control_time,
                "observation_control_time": sample.observation_control_time,
                "lag_rows": sample.lag_rows,
                "lag_seconds": sample.lag_seconds,
                "state_match_rmse": sample.state_match_rmse,
                "state_match_max_abs": sample.state_match_max_abs,
            }
        )
    _atomic_write_text(path, _jsonl(rows))


def load_tasks(tasks_file: Path | None, inline_tasks: list[str] | None) -> list[str]:
    if tasks_file is not None and inline_tasks:
        raise ValueError("use either --tasks-file or --task, not both")
    if tasks_file is not None:
        value = json.loads(tasks_file.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("tasks JSON must contain a list of strings")
        tasks = value
    else:
        tasks = inline_tasks or []
    if not tasks or not all(isinstance(task, str) and task.strip() for task in tasks):
        raise ValueError("at least one non-empty task is required")
    if len(set(tasks)) != len(tasks):
        raise ValueError("task descriptions must be unique")
    return [task.strip() for task in tasks]


def write_dataset_metadata(
    output_root: Path,
    summaries: list[EpisodeSummary],
    tasks: list[str],
    state_names: tuple[str, ...],
    fps: float,
    width: int,
    height: int,
) -> None:
    meta_dir = output_root / "meta"
    _write_json(
        meta_dir / "info.json", build_info(summaries, state_names, fps, width, height, len(tasks))
    )
    _write_json(meta_dir / "modality.json", build_modality())
    _atomic_write_text(
        meta_dir / "tasks.jsonl",
        _jsonl([{"task_index": index, "task": task} for index, task in enumerate(tasks)]),
    )
    _atomic_write_text(
        meta_dir / "episodes.jsonl",
        _jsonl(
            [
                {
                    "episode_index": summary.episode_index,
                    "tasks": [summary.task],
                    "length": summary.frames,
                }
                for summary in summaries
            ]
        ),
    )
    _atomic_write_text(
        meta_dir / "conversion_manifest.jsonl",
        _jsonl([asdict(summary) for summary in summaries]),
    )


def convert_dataset(args: argparse.Namespace) -> None:
    recordings = find_recordings(args.input)
    if args.limit is not None:
        recordings = recordings[: args.limit]
    tasks = load_tasks(args.tasks_file, args.task)
    output_root = args.output.expanduser().resolve()
    data_root = Path(__file__).resolve().parents[1] / "data"
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[EpisodeSummary] = []
    expected_state_names: tuple[str, ...] | None = None
    global_frame_index = 0

    for episode_index, recording in enumerate(recordings):
        print(f"[scan] episode={episode_index} recording={recording.directory.name}")
        scan = scan_recording(
            recording,
            state_match_window_rows=args.state_match_window_rows,
            max_state_match_error=args.max_state_match_error,
            allow_policy_seq_gaps=args.allow_policy_seq_gaps,
        )
        if expected_state_names is None:
            expected_state_names = scan.state_names
        elif scan.state_names != expected_state_names:
            raise ValueError(f"joint names/order changed in {recording.csv_path}")

        chunk_index = episode_index // CHUNK_SIZE
        stem = f"episode_{episode_index:06d}"
        parquet_path = output_root / "data" / f"chunk-{chunk_index:03d}" / f"{stem}.parquet"
        video_path = (
            output_root
            / "videos"
            / f"chunk-{chunk_index:03d}"
            / "observation.images.ego_view"
            / f"{stem}.mp4"
        )
        audit_path = output_root / "audit" / f"chunk-{chunk_index:03d}" / f"{stem}.jsonl"
        summary_path = output_root / "audit" / f"chunk-{chunk_index:03d}" / f"{stem}.summary.json"
        task_index = episode_index % len(tasks)
        if args.resume and (parquet_path.exists() or video_path.exists() or summary_path.exists()):
            required_paths = [parquet_path, video_path, audit_path, summary_path]
            missing_paths = [path for path in required_paths if not path.exists()]
            if missing_paths:
                raise FileExistsError(
                    "cannot resume an incomplete episode; pass --overwrite after inspecting it. "
                    f"Missing: {missing_paths}"
                )
            summary = _read_episode_summary(summary_path)
            if (
                summary.episode_index != episode_index
                or summary.recording_name != recording.directory.name
                or summary.task_index != task_index
                or summary.task != tasks[task_index]
                or summary.frames != len(scan.samples)
                or summary.skipped_boundary_policy_frames != scan.skipped_boundary_policy_frames
            ):
                raise ValueError(f"resume metadata does not match source episode: {summary_path}")
            existing_video = probe_video(
                video_path,
                summary.frames,
                camera_fovy=summary.video.camera_fovy,
                camera_position=summary.video.camera_position,
            )
            if args.camera_fovy is not None and not math.isclose(
                summary.video.camera_fovy, args.camera_fovy, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(
                    f"resume video fovy is {summary.video.camera_fovy}, requested "
                    f"{args.camera_fovy}: {video_path}"
                )
            if args.camera_pos is not None and (
                summary.video.camera_position is None
                or not np.allclose(
                    summary.video.camera_position,
                    args.camera_pos,
                    rtol=0.0,
                    atol=1e-9,
                )
            ):
                raise ValueError(
                    f"resume video camera position is {summary.video.camera_position}, "
                    f"requested {tuple(args.camera_pos)}: {video_path}"
                )
            if existing_video != summary.video:
                raise ValueError(f"resume video metadata changed: {video_path}")
            parquet_rows = _parquet_row_count(parquet_path)
            if parquet_rows != summary.frames:
                raise ValueError(
                    f"resume Parquet has {parquet_rows} rows, expected {summary.frames}: "
                    f"{parquet_path}"
                )
            _write_json(summary_path, asdict(summary))
            summaries.append(summary)
            global_frame_index += summary.frames
            print(f"[resume] episode={episode_index} rows=frames={summary.frames}")
            continue
        if not args.overwrite and (parquet_path.exists() or video_path.exists()):
            raise FileExistsError(
                f"episode output exists; pass --overwrite to replace it: {parquet_path}"
            )

        temporary_parquet = parquet_path.with_name(f".{stem}.partial.parquet")
        temporary_video = video_path.with_name(f".{stem}.partial.mp4")
        temporary_parquet.unlink(missing_ok=True)
        temporary_video.unlink(missing_ok=True)
        try:
            print(f"[render] episode={episode_index} frames={len(scan.samples)}")
            video_info = render_video(
                scan,
                temporary_video,
                camera=args.camera,
                width=args.width,
                height=args.height,
                fps=args.fps,
                crf=args.crf,
                data_root=data_root,
                camera_fovy=args.camera_fovy,
                camera_position=tuple(args.camera_pos) if args.camera_pos is not None else None,
            )
            if video_info.width != args.width or video_info.height != args.height:
                raise ValueError(f"unexpected video dimensions: {video_info}")
            if not math.isclose(video_info.fps, args.fps, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"unexpected video fps: {video_info.fps}")
            if video_info.codec != "h264" or video_info.pixel_format != "yuv420p":
                raise ValueError(f"unexpected video encoding: {video_info}")

            parquet_rows = write_episode_parquet(
                temporary_parquet,
                scan.samples,
                episode_index=episode_index,
                task_index=task_index,
                global_start_index=global_frame_index,
                fps=args.fps,
            )
            if parquet_rows != video_info.frames:
                raise ValueError(
                    f"Parquet/video mismatch: rows={parquet_rows}, frames={video_info.frames}"
                )

            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_parquet, parquet_path)
            os.replace(temporary_video, video_path)
            write_alignment_audit(audit_path, scan.samples)

            summary = EpisodeSummary(
                episode_index=episode_index,
                recording_name=recording.directory.name,
                source_csv=str(recording.csv_path),
                task_index=task_index,
                task=tasks[task_index],
                frames=len(scan.samples),
                first_policy_seq=scan.samples[0].policy_seq,
                last_policy_seq=scan.samples[-1].policy_seq,
                mean_lag_ms=float(np.mean([sample.lag_seconds for sample in scan.samples]) * 1000),
                max_state_match_error=max(sample.state_match_max_abs for sample in scan.samples),
                skipped_boundary_policy_frames=scan.skipped_boundary_policy_frames,
                video=video_info,
            )
            summaries.append(summary)
            _write_json(summary_path, asdict(summary))
            global_frame_index += len(scan.samples)
            print(
                f"[ok] episode={episode_index} rows=frames={len(scan.samples)} "
                f"mean_lag_ms={summary.mean_lag_ms:.3f}"
            )
        finally:
            temporary_parquet.unlink(missing_ok=True)
            temporary_video.unlink(missing_ok=True)

    assert expected_state_names is not None
    write_dataset_metadata(
        output_root,
        summaries,
        tasks,
        expected_state_names,
        args.fps,
        args.width,
        args.height,
    )
    print(f"[done] episodes={len(summaries)} frames={global_frame_index} output={output_root}")


def preflight_dataset(args: argparse.Namespace) -> None:
    recordings = find_recordings(args.input)
    if args.limit is not None:
        recordings = recordings[: args.limit]

    total_frames = 0
    all_lag_seconds: list[float] = []
    maximum_match_error = 0.0
    skipped_boundary_policy_frames = 0
    for episode_index, recording in enumerate(recordings):
        scan = scan_recording(
            recording,
            state_match_window_rows=args.state_match_window_rows,
            max_state_match_error=args.max_state_match_error,
            allow_policy_seq_gaps=args.allow_policy_seq_gaps,
        )
        episode_max_error = max(sample.state_match_max_abs for sample in scan.samples)
        episode_mean_lag_ms = float(np.mean([sample.lag_seconds for sample in scan.samples]) * 1000)
        total_frames += len(scan.samples)
        all_lag_seconds.extend(sample.lag_seconds for sample in scan.samples)
        maximum_match_error = max(maximum_match_error, episode_max_error)
        skipped_boundary_policy_frames += scan.skipped_boundary_policy_frames
        print(
            f"[preflight] {episode_index + 1}/{len(recordings)} "
            f"recording={recording.directory.name} frames={len(scan.samples)} "
            f"mean_lag_ms={episode_mean_lag_ms:.3f} max_error={episode_max_error:.6f} "
            f"skipped_boundary={scan.skipped_boundary_policy_frames}"
        )

    mean_lag_ms = float(np.mean(all_lag_seconds) * 1000)
    print(
        f"[preflight-done] episodes={len(recordings)} frames={total_frames} "
        f"mean_lag_ms={mean_lag_ms:.3f} max_error={maximum_match_error:.6f} "
        f"skipped_boundary={skipped_boundary_policy_frames}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert replay_mujoco_csv data into GR00T LeRobot v2 format."
    )
    parser.add_argument(
        "input", type=Path, help="recording directory, data.csv, or parent directory"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--task", action="append", help="task text; repeat for paraphrases")
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument(
        "--camera-fovy", type=float, help="explicit preview override; defaults to the XML value"
    )
    parser.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="camera position relative to its parent body; defaults to the XML value",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=EXPECTED_FPS)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--state-match-window-rows", type=int, default=12)
    parser.add_argument("--max-state-match-error", type=float, default=0.05)
    parser.add_argument("--allow-policy-seq-gaps", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--preflight-only", action="store_true", help="validate CSV inputs without rendering"
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume", action="store_true", help="validate and skip completed episodes"
    )
    output_mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("width, height, and fps must be positive")
    if not 0 <= args.crf <= 51:
        raise ValueError("crf must be between 0 and 51")
    if args.camera_fovy is not None and not 0 < args.camera_fovy < 180:
        raise ValueError("camera fovy must be between 0 and 180 degrees")
    if args.camera_pos is not None and not all(math.isfinite(value) for value in args.camera_pos):
        raise ValueError("camera position values must be finite")
    if args.preflight_only:
        preflight_dataset(args)
        return 0
    if args.output is None:
        raise ValueError("--output is required unless --preflight-only is used")
    convert_dataset(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
