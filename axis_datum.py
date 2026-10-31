"""Feature-based common-axis datum fitting from probe-selected surface sections."""

import numpy as np

from lathe_axis import fit_lathe_axis
from lathe_contact import surface_contact
from lathe_spindle import (
    SpindleFit,
    contact_profiles,
    correct_spindle_pose,
    first_harmonics,
    pose_matrix,
    sampling_uncertainty,
    spin_matrix,
)


def geometric_axis_step(locator, model_matrix, parameters, probes, extent):
    pose = pose_matrix(parameters)
    points, groups = [], []
    for angle in np.arange(64) * 2 * np.pi / 64:
        part = spin_matrix(angle) @ pose
        inverse = np.linalg.inv(part)
        matrix = part @ model_matrix
        for group, (height, x, y, _radius) in enumerate(probes):
            direction = np.array((x, y, 0.0))
            direction /= np.linalg.norm(direction)
            station = np.array((0.0, 0.0, height))
            contact = surface_contact(
                locator, station + direction * extent, station - direction * extent, matrix
            )
            # A hole is lost contact, not a contact on the opposite/internal wall.
            if contact is None or contact @ direction <= 0:
                continue
            points.append((inverse @ np.r_[contact, 1.0])[:3])
            groups.append(group)
    if any(groups.count(group) < 16 for group in range(len(probes))):
        raise ValueError("More exterior surface coverage needed")
    points, groups = np.asarray(points), np.asarray(groups)
    fitted = fit_lathe_axis(points, np.array((0.0, 0.0, 1.0)), groups)
    axis = fitted.axis if fitted.axis[2] >= 0 else -fitted.axis
    rotation_axis = np.cross(axis, (0.0, 0.0, 1.0))
    angle = np.arccos(np.clip(axis[2], -1.0, 1.0))
    target = np.zeros(4)
    if angle > 1e-10:
        target[2:] = rotation_axis[:2] / max(np.linalg.norm(rotation_axis), 1e-12) * angle
    rotation = pose_matrix(target)[:3, :3]
    target[:2] = -(rotation @ fitted.center)[:2]
    angles, before_profiles = contact_profiles(
        locator, model_matrix, parameters, probes, extent, count=96
    )
    signal = first_harmonics(angles, before_profiles)
    before = np.hypot(signal[::2], signal[1::2]) * 2
    difference = target - parameters
    fraction = min(
        1.0,
        np.radians(2) / max(np.linalg.norm(difference[2:]), 1e-12),
        0.08 * extent / max(np.linalg.norm(difference[:2]), 1e-12),
    )
    rank = 4 if len(probes) > 1 and np.ptp([probe[0] for probe in probes]) > extent * 0.01 else 2
    if rank == 2:
        target[2:] = parameters[2:]
        difference = target - parameters
    for amount in (fraction, fraction * 0.5, fraction * 0.25, fraction * 0.125):
        candidate = parameters + difference * amount
        if np.linalg.norm(candidate[2:]) > np.radians(20):
            continue
        _, profiles = contact_profiles(locator, model_matrix, candidate, probes, extent, count=96)
        after_signal = first_harmonics(angles, profiles)
        after = np.hypot(after_signal[::2], after_signal[1::2]) * 2
        if np.linalg.norm(after) < np.linalg.norm(before) - 1e-10 and np.all(
            after <= before + np.maximum(before * 0.02, extent * 1e-4)
        ):
            return SpindleFit(
                candidate,
                before,
                after,
                np.nanmax(profiles, axis=1) - np.nanmin(profiles, axis=1),
                rank,
                True,
            )
    return SpindleFit(
        np.asarray(parameters).copy(),
        before,
        before,
        np.nanmax(before_profiles, axis=1) - np.nanmin(before_profiles, axis=1),
        rank,
        False,
    )


def fit_probe_datum(locator, model_matrix, parameters, probes, extent):
    """Fit datum features, then refine and validate the measured rotational error."""
    geometric = geometric_axis_step(locator, model_matrix, parameters, probes, extent)
    measured = correct_spindle_pose(locator, model_matrix, parameters, probes, extent)
    # Independent denser, phase-shifted contacts were not used to propose either step.
    angles, profiles = contact_profiles(
        locator, model_matrix, parameters, probes, extent, count=192, phase_offset=.5
    )
    signal = first_harmonics(angles, profiles)
    before = np.hypot(signal[::2], signal[1::2]) * 2
    best = SpindleFit(
        np.asarray(parameters).copy(), before, before,
        np.nanmax(profiles, axis=1) - np.nanmin(profiles, axis=1),
        measured.rank, False, sampling_uncertainty(angles, profiles),
    )
    for candidate in (geometric, measured):
        if not candidate.accepted:
            continue
        _, check = contact_profiles(
            locator, model_matrix, candidate.parameters, probes, extent,
            count=192, phase_offset=.5,
        )
        signal = first_harmonics(angles, check)
        after = np.hypot(signal[::2], signal[1::2]) * 2
        if meaningful_improvement(best.after, after, extent) and np.all(
            after <= before + np.maximum(before * 1e-4, extent * 1e-7)
        ):
            best = SpindleFit(
                candidate.parameters.copy(), before, after,
                np.nanmax(check, axis=1) - np.nanmin(check, axis=1),
                candidate.rank, True, sampling_uncertainty(angles, check),
            )
    return best


def meaningful_improvement(before, after, extent):
    """Require an absolute and relative reduction in the independently measured error."""
    baseline = float(np.linalg.norm(before))
    gain = baseline - float(np.linalg.norm(after))
    return gain > max(extent * 1e-8, baseline * 1e-5)
