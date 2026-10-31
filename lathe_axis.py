"""Robust centerline estimates from a rotating contact-probe sweep."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatheFit:
    center: np.ndarray
    axis: np.ndarray
    radius: float
    runout: float
    coverage: float
    error: float


def fit_lathe_axis(points, initial_axis, groups=None):
    points = np.asarray(points, dtype=float)
    if len(points) < 12 or not np.all(np.isfinite(points)):
        raise ValueError("More surface readings needed")
    groups = np.zeros(len(points), dtype=int) if groups is None else np.asarray(groups, dtype=int)
    if groups.shape != (len(points),):
        raise ValueError("Probe readings do not match their contacts")
    labels, counts = np.unique(groups, return_counts=True)
    usable = np.isin(groups, labels[counts >= 6])
    points, groups = points[usable], groups[usable]
    if len(points) < 12:
        raise ValueError("More surface readings needed")
    _, groups = np.unique(groups, return_inverse=True)
    group_count = int(groups.max()) + 1
    axis = np.asarray(initial_axis, dtype=float)
    axis /= max(np.linalg.norm(axis), 1e-12)
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    right = np.cross(axis, helper)
    right /= np.linalg.norm(right)
    up = np.cross(axis, right)
    origin = np.mean(points, axis=0)
    local = np.column_stack(((points-origin) @ right, (points-origin) @ up, (points-origin) @ axis))
    scale = max(float(np.max(np.ptp(local, axis=0))), 1e-9)
    local /= scale
    design = np.column_stack((2*local[:, 0], 2*local[:, 1], np.eye(group_count)[groups]))
    if np.linalg.cond(design) > 1e5:
        raise ValueError("Sweep a curved surface")
    cx, cy = np.linalg.lstsq(design, np.sum(local[:, :2]**2, axis=1), rcond=None)[0][:2]
    distances = np.linalg.norm(local[:, :2] - (cx, cy), axis=1)
    radii = np.array([np.median(distances[groups == i]) for i in range(group_count)])
    def residual(values):
        center = np.array((values[0], values[1], 0.))
        direction = np.array((values[2], values[3], 1.))
        direction /= np.linalg.norm(direction)
        delta = local-center
        radial = delta - np.outer(delta @ direction, direction)
        return np.linalg.norm(radial, axis=1) - values[4:][groups]
    # A single-height sweep identifies centering, but does not constrain tilt.
    tilt = .36 if np.ptp(local[:, 2]) > .06 else 1e-9
    values = np.r_[cx, cy, 0., 0., radii]
    lower = np.r_[-4., -4., -tilt, -tilt, np.full(group_count, 1e-6)]
    upper = np.r_[4., 4., tilt, tilt, np.full(group_count, 8.)]
    parameter_count = len(values)
    values = np.clip(values, lower, upper)
    def objective(error):
        return np.sum(np.sqrt(1+(error/.005)**2)-1)
    for _ in range(40):
        error = residual(values)
        jacobian = np.column_stack([
            (residual(values + np.eye(parameter_count)[i]*1e-5)-error)/1e-5 for i in range(parameter_count)
        ])
        weights = (1+(error/.005)**2)**-.25
        step = np.linalg.lstsq(
            np.vstack((jacobian*weights[:, None], np.eye(parameter_count)*1e-5)),
            np.r_[-error*weights, np.zeros(parameter_count)], rcond=None,
        )[0]
        if tilt < 1e-8:
            step[2:4] = 0
        score = objective(error)
        for amount in (1., .5, .25, .125, .0625):
            candidate = np.clip(values + step*amount, lower, upper)
            if objective(residual(candidate)) < score:
                values = candidate
                break
        else:
            break
        if np.linalg.norm(step*amount) < 1e-8:
            break
    x, y, u, v = values[:4]
    radius = np.median(values[4:])
    center = origin + scale * (x*right + y*up)
    direction = axis + u*right + v*up
    direction /= np.linalg.norm(direction)
    delta = points-center
    radial = delta - np.outer(delta @ direction, direction)
    angles = np.arctan2(radial @ up, radial @ right)
    coverages = []
    for group in range(group_count):
        ordered = np.sort(angles[groups == group])
        gaps = np.diff(np.r_[ordered, ordered[0]+2*np.pi])
        coverages.append(float(np.degrees(2*np.pi-gaps.max())))
    coverage = min(coverages)
    if coverage < 80:
        raise ValueError("Keep spinning for more coverage")
    errors = residual(values) * scale
    return LatheFit(center, direction, float(radius*scale),
                    float(np.percentile(errors, 95)-np.percentile(errors, 5)),
                    coverage, float(np.sqrt(np.mean(errors**2))))


def probe_errors(points, groups, center, axis):
    points = np.asarray(points, dtype=float)
    groups = np.asarray(groups, dtype=int)
    delta = points-center
    radial = np.linalg.norm(delta-np.outer(delta @ axis,axis),axis=1)
    errors = {}
    for group in np.unique(groups):
        distances = radial[groups == group]
        if len(distances) >= 12:
            residual = np.sort(np.abs(distances-np.median(distances)))
            retained = residual[:max(1,int(len(residual)*.9))]
            errors[int(group)] = float(np.sqrt(np.mean(retained**2)))
    return errors


def refine_lathe_axis(center, axis, fitted, points, groups, protected_groups):
    """Accept a gradual correction only if earlier contact stations stay aligned."""
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    target = fitted.axis if fitted.axis @ axis >= 0 else -fitted.axis
    target_center = fitted.center + target * float((center-fitted.center) @ target)
    before = probe_errors(points,groups,center,axis)
    distance = np.linalg.norm(target_center-center)
    amount = min(.2, max(fitted.radius,1e-8)*.08/max(distance,1e-12))
    for _ in range(8):
        next_center = center+(target_center-center)*amount
        next_axis = axis+(target-axis)*amount
        next_axis /= np.linalg.norm(next_axis)
        after = probe_errors(points,groups,next_center,next_axis)
        if all(after.get(group,0.) <= error*1.02 + max(fitted.radius,1e-8)*1e-4
               for group,error in before.items() if group in protected_groups):
            return next_center,next_axis,set(after)
        amount *= .5
    return center.copy(),axis.copy(),set(before)
