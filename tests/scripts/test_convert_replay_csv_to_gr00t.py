from collections import deque

import numpy as np
from scripts.convert_replay_csv_to_gr00t import (
    StateSnapshot,
    build_modality,
    compute_projected_gravity,
    find_best_observation_snapshot,
    qpos_to_policy_state,
)


def test_qpos_to_policy_state_reorders_body_and_hands():
    qpos = np.arange(57, dtype=np.float64)

    state = qpos_to_policy_state(qpos)

    expected = np.concatenate([qpos[7:29], qpos[36:43], qpos[29:36], qpos[43:50]])
    np.testing.assert_array_equal(state, expected.astype(np.float32))
    assert state.shape == (43,)


def test_compute_projected_gravity_uses_inverse_wxyz_rotation():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(compute_projected_gravity(identity), [0.0, 0.0, -1.0])

    half_angle = np.pi / 4
    body_to_world_roll_90 = np.array([np.cos(half_angle), np.sin(half_angle), 0.0, 0.0])
    np.testing.assert_allclose(
        compute_projected_gravity(body_to_world_roll_90),
        [0.0, -1.0, 0.0],
        atol=1e-6,
    )


def test_find_best_observation_snapshot_selects_matching_previous_row():
    current_qpos = np.arange(57, dtype=np.float64)
    previous_qpos = current_qpos.copy()
    previous_qpos[7:50] += 0.25
    snapshots = deque(
        [
            StateSnapshot(10, 0.025, 1.025, previous_qpos),
            StateSnapshot(11, 0.0275, 1.0275, current_qpos),
        ]
    )
    received_state = qpos_to_policy_state(previous_qpos)

    best, rmse, max_abs = find_best_observation_snapshot(snapshots, received_state)

    assert best.row_index == 10
    assert rmse == 0.0
    assert max_abs == 0.0


def test_modality_matches_builtin_unitree_g1_sonic_keys():
    modality = build_modality()

    assert list(modality["state"]) == [
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "projected_gravity",
    ]
    assert list(modality["action"]) == [
        "motion_token",
        "left_hand_joints",
        "right_hand_joints",
    ]
    assert modality["video"]["ego_view"]["original_key"] == "observation.images.ego_view"
