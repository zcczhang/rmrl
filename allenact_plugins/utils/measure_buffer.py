from typing import Optional, Dict

import numpy as np
from fastdtw import fastdtw

from allenact_plugins.utils.env_utils import dict_space_to_gym_Box_space
from allenact_plugins.utils.metics import states_measure, states_entropy

__all__ = ["MeasureBuffer"]


class MeasureBuffer:
    def __init__(self, fixed_history_length: Optional[int] = None):
        self._history: Optional[dict] = None
        self.fixed_history_length = fixed_history_length
        self._current_history_length = 0

    @property
    def history(self):
        return self._history

    @property
    def history_length(self):
        return self._current_history_length

    def clear_history(self):
        self._history = None
        self._current_history_length = 0

    def extend_single_obs(self, obs_dict: Dict[str, np.ndarray]):
        if self._history is None:
            self._history = {
                k: np.array(v).reshape(1, np.array(v).shape[-1])
                for k, v in obs_dict.items()
            }
            self._current_history_length = len(list(self._history.values())[0])
            assert self._current_history_length == 1, self._history
            return
        cur_history_length = self._current_history_length
        for k, v in obs_dict.items():
            assert k in self._history, f"{k} not in {self._history.keys()}"
            axis = 0
            if self.fixed_history_length is not None:
                assert (
                    cur_history_length <= self.fixed_history_length
                ), cur_history_length
                if cur_history_length == self.fixed_history_length:
                    axis = 1
            self._history[k] = np.concatenate(
                [self._history[k][axis:, ...], [v]], axis=0
            )
        self._current_history_length = len(list(self._history.values())[0])

    def extend_histories(self, histories: Dict[str, np.ndarray]):
        if self._history is None:
            self._history = {
                k: np.array(v)[-self.fixed_history_length or len(v) :, ...]
                for k, v in histories.items()
            }
            self._current_history_length = len(list(self._history.values())[0])
            return
        for k, v in histories.items():
            assert k in self._history, f"{k} not in {self._history.keys()}"
            self._history[k] = np.vstack([self._history[k], v])
            if (
                self.fixed_history_length is not None
                and len(self._history[k]) >= self.fixed_history_length
            ):
                self._history[k] = self._history[k][-self.fixed_history_length :, ...]
        self._current_history_length = len(list(self._history.values())[0])

    def dispersion_measure(
        self, measure_method: str, merge_histories: bool = False, **kwargs
    ) -> Optional[dict]:
        if (
            self.fixed_history_length is not None
            and self.history_length < self.fixed_history_length
        ):
            return None
        if merge_histories and len(self._history.keys()) > 1:
            history = {
                "".join([_ + "_and_" for _ in self._history.keys()])[
                    :-5
                ]: np.concatenate([v for v in self._history.values()], axis=1)
            }
        else:
            history = self.history
        metrics = {}
        for key, key_history in history.items():
            if measure_method == "std":
                k_measure = states_measure(
                    key_history,
                    metrics="std",
                    mean_xyz=True,
                    normalize_first=kwargs.get("normalize_first", False),
                )
            elif "entropy" in measure_method:
                grid_size = kwargs.get("grid_size", None)
                entropy_fn = kwargs.get("entropy_fn", "scipy")
                if entropy_fn == "scipy":
                    assert "world_space" in kwargs
                    space = kwargs["world_space"]
                    space = dict_space_to_gym_Box_space(space.to_dict())
                else:
                    space = None
                k_measure = states_entropy(
                    key_history,
                    entropy_fn=entropy_fn,
                    grid_size=grid_size,
                    world_space=space,
                )
            else:
                raise NotImplementedError(measure_method)
            metrics[key] = k_measure
        return metrics

    def distance_measure(
        self,
        measure_method: str,
        merge_histories: bool = True,
        *,
        mem_steps: Optional[int] = None,
        measure_steps: int,
        normalize_first: bool = False,
    ) -> Optional[dict]:
        if (
            self.fixed_history_length is not None
            and self.history_length < self.fixed_history_length
        ):
            return None
        if self.fixed_history_length is not None and mem_steps is None:
            mem_steps = self.fixed_history_length - measure_steps
        if self.history_length < mem_steps + measure_steps:
            return None
        if merge_histories and len(self._history.keys()) > 1:
            history = {
                "".join([_ + "_and_" for _ in self._history.keys()])[
                    :-5
                ]: np.concatenate([v for v in self._history.values()], axis=1)
            }
        else:
            history = self.history
        metrics = {}
        for key, key_history in history.items():
            if normalize_first:
                key_history = (key_history - np.mean(key_history, axis=0)) / np.std(
                    key_history, axis=0
                )
                key_history = np.nan_to_num(key_history, nan=0.0)
            distances = []
            for i in range(measure_steps):
                p = key_history[mem_steps + i]
                past_states = key_history[: mem_steps - measure_steps + i]
                if measure_method == "euclidean":
                    min_distance_to_past_states = np.min(
                        np.linalg.norm(p - past_states, axis=-1)
                    )
                elif measure_method == "dtw":
                    distances_ = []
                    for past_state in past_states:
                        distance, _ = fastdtw(p, past_state)
                        distances_.append(distance)
                    # min_distance_to_past_states = np.mean(distances_)
                    min_distance_to_past_states = np.min(distances_)
                else:
                    raise NotImplementedError(measure_method)
                distances.append(min_distance_to_past_states)
            maxmin_dist = np.max(distances)
            metrics[key] = maxmin_dist
        return metrics
