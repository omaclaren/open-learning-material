"""Last-observed-box tracker for the Basic tracking chapter.

Uses the last observed box without motion prediction. Geometry follows the
Evaluation chapter. Requires NumPy and SciPy.
Track IDs are local to each run, not supplied annotation IDs.
"""
from copy import deepcopy
import csv

import numpy as np
from scipy.optimize import linear_sum_assignment


def box_area(box):
    x_min, y_min, x_max, y_max = box

    width = x_max - x_min
    height = y_max - y_min

    return width * height


def intersection_area(box_a, box_b):
    intersection_x_min = max(box_a[0], box_b[0])
    intersection_y_min = max(box_a[1], box_b[1])
    intersection_x_max = min(box_a[2], box_b[2])
    intersection_y_max = min(box_a[3], box_b[3])

    intersection_width = max(0, intersection_x_max - intersection_x_min)
    intersection_height = max(0, intersection_y_max - intersection_y_min)

    return intersection_width * intersection_height


def union_area(box_a, box_b):
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    overlap = intersection_area(box_a, box_b)

    return area_a + area_b - overlap


def box_iou(box_a, box_b):
    overlap = intersection_area(box_a, box_b)
    combined_area = union_area(box_a, box_b)

    if combined_area == 0:
        raise ValueError("IoU is undefined when both boxes have zero area")

    return overlap / combined_area


def pairwise_iou(reference_boxes, current_boxes):
    ious = np.zeros(
        (len(reference_boxes), len(current_boxes)),
        dtype=float,
    )

    for row, reference_box in enumerate(reference_boxes):
        for column, current_box in enumerate(current_boxes):
            ious[row, column] = box_iou(
                reference_box, current_box
            )

    return ious


def load_detections(csv_path, minimum_confidence=0.50):
    """Read the supplied post-NMS CSV; apply confidence filtering only.

    Return a dictionary mapping frame numbers to lists of detection records.
    A/B/C labels reproduce the worked example for frames 50/51/52. Other
    frames use F<frame>D<rank>; all these labels are frame-local, not track IDs.
    Frames without retained detections may be absent from the dictionary.
    """
    by_frame = {}
    with open(csv_path, newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream)):
            confidence = float(row["confidence"])
            if confidence < minimum_confidence:
                continue
            frame = int(row["frame"])
            observation = {
                "confidence": confidence,
                "box_xyxy_px": [float(row[key]) for key in
                                ["x_min", "y_min", "x_max", "y_max"]],
                "source_row_zero_based": row_number,
            }
            by_frame.setdefault(frame, []).append(observation)
    for frame, detections in by_frame.items():
        detections.sort(key=lambda d: (-d["confidence"], d["source_row_zero_based"]))
        prefix = {50: "A", 51: "B", 52: "C"}.get(frame, f"F{frame}D")
        for rank, detection in enumerate(detections, start=1):
            detection["label"] = f"{prefix}{rank}"
    return by_frame


def new_tracker(first_frame):
    """Empty state immediately before processing first_frame."""
    return {"frame": first_frame - 1, "next_id": 1, "tracks": {}}


def update_tracks(state, detections, frame, minimum_iou=0.30, inactive_after=3):
    """Return the updated state and an association trace; leave inputs unchanged.

    Process every frame, in order, even when detections is empty. Records
    contain observed boxes, not predicted positions. On the third consecutive
    miss (by default), deactivate the track for subsequent assignments, but
    retain its history. Never reuse an ID or automatically revive an archive.
    Valid finite, positive-area, correctly ordered xyxy boxes are assumed.
    """
    if frame != state["frame"] + 1:
        raise ValueError("Process every frame in sequence, including empty frames")
    if not 0 <= minimum_iou <= 1:
        raise ValueError("minimum_iou must be between 0 and 1")
    if not isinstance(inactive_after, int) or inactive_after < 1:
        raise ValueError("inactive_after must be a positive integer")

    result = deepcopy(state)
    tracks = result["tracks"]
    active_ids = [tid for tid, track in tracks.items() if track["active"]]
    reference_boxes = [tracks[tid]["last_observed_box_xyxy_px"] for tid in active_ids]
    current_boxes = [detection["box_xyxy_px"] for detection in detections]
    ious = pairwise_iou(reference_boxes, current_boxes)

    rows, columns = linear_sum_assignment(-ious)
    accepted = []
    for row, column in zip(rows, columns):
        if ious[row, column] >= minimum_iou:
            accepted.append((int(row), int(column)))

    by_track = {active_ids[row]: column for row, column in accepted}
    newly_inactive = []
    for tid in active_ids:
        track = tracks[tid]
        if tid in by_track:
            observation = deepcopy(detections[by_track[tid]])
            track["observations"][frame] = observation
            track["last_observed_frame"] = frame
            track["last_observed_label"] = observation["label"]
            track["last_observed_box_xyxy_px"] = deepcopy(observation["box_xyxy_px"])
            track["missed_frames"] = 0
        else:
            track["observations"][frame] = None
            track["missed_frames"] += 1
            if track["missed_frames"] >= inactive_after:
                track["active"] = False
                newly_inactive.append(tid)

    used_detections = {column for _, column in accepted}
    new_tracks = []
    for column, detection in enumerate(detections):
        if column in used_detections:
            continue
        tid = f"T{result['next_id']}"
        result["next_id"] += 1
        observation = deepcopy(detection)
        tracks[tid] = {
            "track_id": tid,
            "observations": {frame: observation},
            "last_observed_frame": frame,
            "last_observed_label": observation["label"],
            "last_observed_box_xyxy_px": deepcopy(observation["box_xyxy_px"]),
            "active": True,
            "missed_frames": 0,
        }
        new_tracks.append(tid)

    result["frame"] = frame
    trace = {
        "frame": frame,
        "row_track_ids": active_ids,
        "reference_boxes": reference_boxes,
        "reference_frames": [state["tracks"][tid]["last_observed_frame"] for tid in active_ids],
        "matrix_shape": list(ious.shape),
        "iou_matrix": ious.tolist(),
        "candidate_pairs": [[active_ids[row], detections[column]["label"], float(ious[row, column])]
                            for row, column in zip(rows, columns)],
        "accepted_pairs": [[active_ids[row], detections[column]["label"]]
                           for row, column in accepted],
        "new_tracks": new_tracks,
        "newly_inactive": newly_inactive,
    }
    return result, trace
