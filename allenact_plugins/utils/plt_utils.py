from typing import List, Union, Any

import cv2
import numpy as np

__all__ = ["round_metrics", "debug_texts_to_frame", "add_boundary_from_success"]


def round_metrics(metrics: Any, n: int = 4) -> Any:
    if isinstance(metrics, (float, int)):
        return round(metrics, n)
    elif isinstance(metrics, dict):
        return {k: round_metrics(v) for k, v in metrics.items()}
    elif isinstance(metrics, (list, tuple)):
        return type(metrics)([round_metrics(m) for m in metrics])
    else:
        return metrics


def debug_texts_to_frame(
    frame: np.ndarray, debug_text: Union[List[str], dict], **kwargs
) -> np.ndarray:
    if isinstance(debug_text, dict):
        text_list = [f"{k}: {v}" for k, v in debug_text.items()]
    else:
        text_list = debug_text
    org_x_init = kwargs.pop("org_x_init", 10)
    org_x_increment = kwargs.pop("org_x_increment", 0)
    org_y_init = kwargs.pop("org_y_init", 30)
    org_y_increment = kwargs.pop("org_y_increment", 20)

    cv2_kwargs = dict(
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0, 255, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    cv2_kwargs.update(kwargs)
    for i, txt in enumerate(text_list):
        cv2.putText(
            frame,
            txt,
            (org_x_init + i * org_x_increment, org_y_init + i * org_y_increment),
            **cv2_kwargs,
        )
    return frame


def add_boundary_from_success(
    frame: np.ndarray,
    success: bool,
    padding: int = 5,
    success_color: tuple = (0, 255, 0),
    fail_color: tuple = (255, 0, 0),
) -> np.ndarray:
    color = np.array(success_color) if success else np.array(fail_color)
    h, w, c = frame.shape
    new_h, new_w = h + 2 * padding, w + 2 * padding
    new_frame = np.full((new_h, new_w, c), color, dtype=np.uint8)
    new_frame[padding:-padding, padding:-padding] = frame
    return new_frame
