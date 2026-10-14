import numpy as np


def cumulative_path(world_pos):
    """Distance travelled to reach each sample"""
    pos = np.asarray(world_pos, dtype=float)
    cum = np.zeros(len(pos), dtype=float)
    if len(pos) > 1:
        np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1), out=cum[1:])
    return cum


def contact_times(wall_hits):
    """Times where the step meets any wall. One step that clips two faces is one contact."""
    hits = np.asarray(wall_hits)
    return np.flatnonzero(np.any(hits != 0, axis=1))


def _segment_length(cumpath, t0, t1):
    """Path length of samples ``[t0, t1)``."""
    if t1 <= t0:
        return 0.0
    return float(cumpath[t1 - 1] - cumpath[t0])


def fit_flight_plane(positions):
    """Finds the best fitting plane to the flight positions, return the axises and the noraml """
    pos = np.asarray(positions, dtype=float)
    #subtract the mean position to centre the data
    centred = pos - pos.mean(axis=0)
    #perform SVD on the centred data
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    if vt.shape[0] < 3:
        raise ValueError("the flight positions do not span a plane")
    e1 = vt[0].copy()
    e2 = vt[1].copy()
    n = vt[2].copy()
    # SVD may return a left-handed frame
    if float(np.dot(np.cross(e1, e2), n)) < 0.0:
        e2 = -e2
    # Point the normal into the +z hemisphere. Flip e2 with it to keep the hand.
    if _negative_hemisphere(n):
        n = -n
        e2 = -e2
    return e1, e2, n


def _negative_hemisphere(n):
    if n[2] < 0.0:
        return True
    if n[2] == 0.0 and n[1] < 0.0:
        return True
    return bool(n[2] == 0.0 and n[1] == 0.0 and n[0] < 0.0)


def label_wall_flights(traj, grid_spacing):
    """Label each scored flight and store the closest plane of the flight, stored as a plane normal"""
    grid_spacing = float(grid_spacing)
    pos = np.asarray(traj.world_pos, dtype=float)
    # Distance travelled to reach each sample
    cum = cumulative_path(pos)
    #number of times the step meets any wall
    hits = [int(t) for t in contact_times(traj.wall_hits)]

    #label each long enough flight and store its plane normal
    flight_of = np.full(len(pos), -1, dtype=int)
    flight_table = []
    route_table = []
    n_under = 0
    #loop over times the trajectory meets a wall
    for t0, t1 in zip(hits, hits[1:]):
        #path length of the flight
        length_m = _segment_length(cum, t0, t1)
        #if the flight is shorter than one grid period, it is not scored
        if length_m < grid_spacing:
            n_under += 1
            continue
        #if the flight has fewer than 3 samples, it is not scored
        if t1 - t0 < 3:
            raise ValueError(
                f"flight [{t0}, {t1}) is {length_m:.3f} m but has fewer than 3 samples"
            )
        
        #fit the plane of the flight
        e1, e2, n = fit_flight_plane(pos[t0:t1])
        flight_id = len(flight_table)
        flight_of[t0:t1] = flight_id
        flight_table.append(dict(
            flight=flight_id, route=flight_id, t0=t0, t1=t1,
        ))
        route_table.append(dict(
            route=flight_id,
            n_fit=n,
            basis=np.column_stack((e1, e2)),
        ))

    traj.flight_of = flight_of
    traj.flight_table = flight_table
    traj.route_table = route_table
    return dict(
        n_contacts=int(len(hits)),
        n_under_period=int(n_under),
        n_scored=int(len(flight_table)),
    )


def repeat_labeled_flights(traj, n_repeats):
    """ akes a walk that has already been split into wall to wall lights and lays that same walk down n_repeats times
    """
    n = int(n_repeats)
    if n < 1:
        raise ValueError("n_repeats must be at least 1")
    table = list(traj.flight_table or [])
    n_flights = len(table)
    if n == 1 or n_flights == 0:
        return dict(n_repeats=n, n_passes=n * n_flights)

    pos = np.asarray(traj.world_pos)
    T = len(pos)
    base_of = np.asarray(traj.flight_of)
    parts_of = []
    repeated = []
    for i in range(n):
        shifted = base_of.copy()
        hit = shifted >= 0
        shifted[hit] += i * n_flights
        parts_of.append(shifted)
        for row in table:
            repeated.append(dict(
                flight=int(row["flight"]) + i * n_flights,
                route=int(row["route"]),
                t0=int(row["t0"]) + i * T,
                t1=int(row["t1"]) + i * T,
            ))

    traj.flight_of = np.concatenate(parts_of)
    traj.flight_table = repeated
    traj.world_pos = np.concatenate([pos] * n, axis=0)
    traj.v_body_seq = np.concatenate([np.asarray(traj.v_body_seq)] * n, axis=0)
    traj.torus_gt = np.concatenate([np.asarray(traj.torus_gt)] * n, axis=0)
    if traj.n_true_seq is not None:
        traj.n_true_seq = np.concatenate(
            [np.asarray(traj.n_true_seq)] * n, axis=0)
    if traj.wall_hits is not None:
        traj.wall_hits = np.concatenate(
            [np.asarray(traj.wall_hits)] * n, axis=0)
    return dict(n_repeats=n, n_passes=n * n_flights)
