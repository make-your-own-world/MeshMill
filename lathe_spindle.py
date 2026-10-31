"""Fixed-spindle centering and levelling from multi-height contact traces."""

from dataclasses import dataclass

import numpy as np

from lathe_contact import surface_contact


def spin_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        ((c, -s, 0.0, 0.0), (s, c, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    )


def pose_matrix(parameters):
    x, y, rx, ry = parameters
    vector = np.array((rx, ry, 0.0))
    angle = np.linalg.norm(vector)
    rotation = np.eye(3)
    if angle > 1e-14:
        axis = vector / angle
        a, b, c = axis
        cross = np.array(((0.0, -c, b), (c, 0.0, -a), (-b, a, 0.0)))
        rotation += np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:2, 3] = (x, y)
    return matrix


@dataclass(frozen=True)
class SpindleFit:
    parameters: np.ndarray
    before: np.ndarray
    after: np.ndarray
    total_runout: np.ndarray
    rank: int
    accepted: bool
    noise_floor: np.ndarray | None = None


def contact_profiles(locator, model_matrix, parameters, probes, extent, count=48, phase_offset=0.):
    angles = (np.arange(count) + phase_offset) * 2 * np.pi / count
    profiles = np.full((len(probes), count), np.nan)
    pose = pose_matrix(parameters)
    for index, angle in enumerate(angles):
        matrix = spin_matrix(angle) @ pose @ model_matrix
        for group, (height, lateral, front, _radius) in enumerate(probes):
            radial = np.array((lateral, front, 0.0))
            radial /= np.linalg.norm(radial)
            station = np.array((0.0, 0.0, height))
            hit = surface_contact(
                locator, station + radial * extent, station - radial * extent, matrix
            )
            if hit is not None and float(hit @ radial) > 0:
                profiles[group, index] = hit @ radial
    return angles, profiles


def first_harmonics(angles, profiles):
    residual = []
    for readings in profiles:
        valid = np.isfinite(readings)
        if np.sum(valid) < max(12, len(angles) * 0.8):
            raise ValueError("Probe loses contact during the turn")
        a = angles[valid]
        # Keep ovality and higher-order shape out of the centering signal.
        design = np.column_stack(
            (
                np.ones(len(a)),
                np.cos(a),
                np.sin(a),
                np.cos(2 * a),
                np.sin(2 * a),
                np.cos(3 * a),
                np.sin(3 * a),
            )
        )
        coefficients = np.linalg.lstsq(design, readings[valid], rcond=None)[0]
        residual.extend(coefficients[1:3])
    return np.asarray(residual)


def sampling_uncertainty(angles,profiles):
    even=first_harmonics(angles[::2],profiles[:,::2])
    odd=first_harmonics(angles[1::2],profiles[:,1::2])
    delta=even-odd
    return 2*np.hypot(delta[::2],delta[1::2])


def correct_spindle_pose(locator, model_matrix, parameters, probes, extent):
    parameters = np.asarray(parameters, dtype=float)
    angles, profiles = contact_profiles(locator, model_matrix, parameters, probes, extent)
    residual = first_harmonics(angles, profiles)
    initial = np.hypot(residual[::2], residual[1::2]) * 2
    steps = np.array((extent * 1e-5, extent * 1e-5, 1e-5, 1e-5))
    heights = np.array([probe[0] for probe in probes])
    active = 4 if len(probes) > 1 and np.ptp(heights) > extent * 0.01 else 2
    jacobian = []
    for column in range(active):
        changed = parameters.copy()
        changed[column] += steps[column]
        sample_angles, samples = contact_profiles(locator, model_matrix, changed, probes, extent)
        jacobian.append((first_harmonics(sample_angles, samples) - residual) / steps[column])
    jacobian = np.column_stack(jacobian)
    # Normalize translations and tilts by their physical lever arm.
    scale = np.array((extent, extent, 1.0, 1.0))[:active]
    normalized = jacobian * scale
    rank = np.linalg.matrix_rank(normalized, tol=max(np.linalg.norm(normalized), 1.0) * 1e-7)
    if rank < active:
        raise ValueError("These surfaces do not determine a unique axis")
    step = np.linalg.lstsq(normalized, -residual, rcond=1e-7)[0] * scale
    limit = min(1.0, 0.08 * extent / max(np.linalg.norm(step[:2]), 1e-12))
    if active == 4:
        limit = min(limit, np.radians(2) / max(np.linalg.norm(step[2:]), 1e-12))
    baseline = np.linalg.norm(initial)
    for fraction in (limit, limit * 0.5, limit * 0.25, limit * 0.125):
        candidate = parameters.copy()
        candidate[:active] += step * fraction
        candidate_angles, candidate_profiles = contact_profiles(
            locator, model_matrix, candidate, probes, extent
        )
        if np.linalg.norm(candidate[2:]) > np.radians(20):
            continue
        corrected = first_harmonics(candidate_angles, candidate_profiles)
        amplitudes = np.hypot(corrected[::2], corrected[1::2]) * 2
        # Evaluate every contact jointly rather than locking a partial alignment.
        improves_every_contact = np.all(amplitudes <= initial + np.maximum(initial*.02,extent*1e-4))
        if np.linalg.norm(amplitudes) < baseline - 1e-10 and improves_every_contact:
            # Check on a denser sweep before changing the preview pose.
            validation_angles,validation_before = contact_profiles(
                locator,model_matrix,parameters,probes,extent,count=96)
            _,validation_after = contact_profiles(locator,model_matrix,candidate,probes,extent,count=96)
            previous_signal = first_harmonics(validation_angles,validation_before)
            next_signal = first_harmonics(validation_angles,validation_after)
            previous_amplitude = np.hypot(previous_signal[::2],previous_signal[1::2])*2
            next_amplitude = np.hypot(next_signal[::2],next_signal[1::2])*2
            if (np.linalg.norm(next_amplitude) >= np.linalg.norm(previous_amplitude)
                    or not np.all(next_amplitude<=previous_amplitude+np.maximum(previous_amplitude*.02,extent*1e-4))):
                continue
            return SpindleFit(
                candidate,
                initial,
                amplitudes,
                np.nanmax(candidate_profiles, axis=1) - np.nanmin(candidate_profiles, axis=1),
                rank,
                True,
            )
    return SpindleFit(
        parameters.copy(),
        initial,
        initial,
        np.nanmax(profiles, axis=1) - np.nanmin(profiles, axis=1),
        rank,
        False,
    )
