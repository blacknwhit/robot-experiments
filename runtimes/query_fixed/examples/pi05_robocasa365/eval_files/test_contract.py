"""Fast contract tests for the PI0.5 RoboCasa365 adapter."""

from __future__ import annotations

import numpy as np

from examples.pi05_robocasa365.eval_files.model2pi05_robocasa365_interface import (
    ACTION_SLICES,
    POLICY_RNG_MODE,
    POLICY_RNG_SEED_KEY,
    POLICY_SEED_VERSION,
    STRICT_DETERMINISM_VERSION,
    PolicyWrapper,
    convert_to_uint8,
    resize_with_pad,
    stable_request_seed,
)


class FakeClient:
    def __init__(self, *, nan: bool = False, native: bool = False, strict: bool = False) -> None:
        self.requests = []
        self.nan = nan
        self.native = native
        self.strict = strict

    def get_server_metadata(self):
        metadata = {
            "test": True,
            "policy_rng_mode": POLICY_RNG_MODE,
            "policy_seed_version": POLICY_SEED_VERSION,
        }
        if self.native:
            metadata["action_output_space"] = "environment"
        if self.strict:
            metadata.update(
                strict_determinism=True,
                determinism_version=STRICT_DETERMINISM_VERSION,
            )
        return metadata

    def predict_action(self, element):
        self.requests.append(element)
        actions = np.arange(50 * 12, dtype=np.float32).reshape(50, 12)
        if self.nan:
            actions[0, 0] = np.nan
        if self.native:
            return {"status": "ok", "data": {"actions": actions[None]}}
        return {"actions": actions}


def make_observations(batch_size: int = 2):
    observations = {
        "annotation.human.task_description": np.asarray([f"task {i}" for i in range(batch_size)]),
    }
    for index, key in enumerate(
        (
            "video.robot0_agentview_left",
            "video.robot0_agentview_right",
            "video.robot0_eye_in_hand",
        )
    ):
        observations[key] = np.full((batch_size, 1, 224, 224, 3), index + 1, dtype=np.uint8)

    dims = {
        "state.end_effector_position_relative": 3,
        "state.end_effector_rotation_relative": 4,
        "state.base_position": 3,
        "state.base_rotation": 4,
        "state.gripper_qpos": 2,
    }
    cursor = 0
    for key, dim in dims.items():
        values = np.arange(cursor, cursor + dim, dtype=np.float32)
        observations[key] = np.tile(values[None, None, :], (batch_size, 1, 1))
        cursor += dim
    return observations


def main() -> None:
    # Exercise the same dtype branches as official openpi-client.image_tools.
    for image in (
        np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3),
        np.linspace(0.0, 1.0, 10 * 20 * 3, dtype=np.float32).reshape(10, 20, 3),
        np.linspace(0.0, 255.0, 10 * 20 * 3, dtype=np.float32).reshape(10, 20, 3),
    ):
        resized = resize_with_pad(image, 16, 16)
        assert resized.shape == (16, 16, 3)
        assert convert_to_uint8(resized).dtype == np.uint8

    observations = make_observations()
    client = FakeClient()
    wrapper = PolicyWrapper(
        n_action_steps=5,
        task_name="robocasa/TestTask",
        base_seed=7,
        client=client,
    )
    result = wrapper.step(observations)["actions"]

    assert len(client.requests) == 2
    assert np.array_equal(client.requests[0]["observation/state"], np.arange(16, dtype=np.float32))
    assert client.requests[0]["prompt"] == "task 0"
    assert client.requests[0][POLICY_RNG_SEED_KEY] == stable_request_seed(
        7, "robocasa/TestTask", 0, 0, 0
    )
    assert client.requests[1][POLICY_RNG_SEED_KEY] == stable_request_seed(
        7, "robocasa/TestTask", 1, 0, 0
    )
    assert np.all(client.requests[0]["observation/image"] == 1)
    assert np.all(client.requests[0]["observation/wrist_image"] == 3)
    assert np.all(client.requests[0]["observation/right_image"] == 2)

    raw = np.arange(50 * 12, dtype=np.float32).reshape(50, 12)[:5]
    for key, (start, end) in ACTION_SLICES.items():
        assert result[key].shape == (2, 5, end - start)
        assert np.array_equal(result[key][0], raw[:, start:end])

    native_client = FakeClient(native=True)
    native_result = PolicyWrapper(
        n_action_steps=5,
        task_name="robocasa/TestTask",
        base_seed=7,
        client=native_client,
    ).step(observations)["actions"]
    assert len(native_client.requests) == 2
    assert set(native_client.requests[0]) >= {"examples"}
    native_example = native_client.requests[0]["examples"][0]
    assert [image.shape for image in native_example["image"]] == [(224, 224, 3)] * 3
    assert native_example["state"].shape == (16,)
    for key, (start, end) in ACTION_SLICES.items():
        assert np.array_equal(native_result[key][0], raw[:, start:end])

    wrapper.finish_step([False, True])
    wrapper.step(observations)
    assert client.requests[2][POLICY_RNG_SEED_KEY] == stable_request_seed(
        7, "robocasa/TestTask", 0, 0, 1
    )
    assert client.requests[3][POLICY_RNG_SEED_KEY] == stable_request_seed(
        7, "robocasa/TestTask", 1, 1, 0
    )

    strict_hashes = []
    for _ in range(2):
        strict_wrapper = PolicyWrapper(
            n_action_steps=5,
            task_name="robocasa/TestTask",
            base_seed=7,
            strict_determinism=True,
            trace_determinism=True,
            client=FakeClient(native=True, strict=True),
        )
        strict_wrapper.step(observations)
        strict_wrapper.finish_step([True, True])
        strict_hashes.append(strict_wrapper.episode_trace_hashes)
    assert strict_hashes[0] == strict_hashes[1]
    assert len(strict_hashes[0]) == 2

    try:
        PolicyWrapper(strict_determinism=True, client=FakeClient(native=True))
    except RuntimeError as exc:
        assert "strict deterministic" in str(exc)
    else:
        raise AssertionError("Non-strict server was accepted by strict client")

    try:
        PolicyWrapper(n_action_steps=5, client=FakeClient(nan=True)).step(observations)
    except ValueError as exc:
        assert "NaN or Inf" in str(exc)
    else:
        raise AssertionError("NaN actions were not rejected")

    print("PI0.5 RoboCasa365 adapter contract tests passed.")


if __name__ == "__main__":
    main()
