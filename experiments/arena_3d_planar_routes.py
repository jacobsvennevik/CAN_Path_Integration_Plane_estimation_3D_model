from dataclasses import dataclass

import numpy as np

from config import ExperimentConfig
from experiments.base import (
    BaseExperiment, Trajectory, torus_gt_from_velocity, walk_params,
)
from model.path_integration import frame

HOME_PERCH = np.array([-1.0, 0.0, 0.0])


def _normal_at(tilt_deg, azimuth):
    """builds the unit normal of one tilted plane."""
    tilt = np.deg2rad(float(tilt_deg))
    azimuth = float(azimuth)
    n = np.array([
        np.sin(tilt) * np.cos(azimuth),
        np.sin(tilt) * np.sin(azimuth),
        np.cos(tilt),
    ])
    return n / np.linalg.norm(n)


def takeoff_heading(n, wall_in, rng, spread_deg):
    """Heading in the plane, within ±spread of the into-box direction."""
    n = np.asarray(n, dtype=float)
    wall = np.asarray(wall_in, dtype=float)
    projected = wall - float(np.dot(wall, n)) * n
    norm = float(np.linalg.norm(projected))
    if norm < 1e-12:
        raise RuntimeError("the take-off wall is parallel to the plane")
    projected = projected / norm
    uv = frame(n).T @ projected
    centre = float(np.arctan2(uv[1], uv[0]))
    spread = np.deg2rad(float(spread_deg))
    return centre + float(rng.uniform(-spread, spread))


def _land_on_face(x, v, limit):
    """Shorten ``v`` so ``x + kept`` ends on the first face the step crosses."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    fracs = []
    for axis in range(3):
        if v[axis] > 1e-15 and x[axis] + v[axis] > limit:
            fracs.append(((limit - x[axis]) / v[axis], axis, 1.0))
        elif v[axis] < -1e-15 and x[axis] + v[axis] < -limit:
            fracs.append(((-limit - x[axis]) / v[axis], axis, -1.0))
    if not fracs:
        return None
    frac, axis, sign = min(fracs, key=lambda item: item[0])
    frac = float(np.clip(frac, 0.0, 1.0))
    landed = np.clip(x + frac * v, -limit, limit)
    landed[axis] = sign * limit
    return landed


def fly_until_wall(x0, n, heading, speed, step_turn_std, limit, rng,
                   max_len_m=20.0):
    """The 2D persistent walk on the plane stopps when the agent hits a wall. 
       Turns a little bit at each step. Result is the stored path.
    """
    #Get the frame of the picked plane
    E = frame(n)
    #Start the walk at the home perch
    x = np.asarray(x0, dtype=float).copy()
    #Path rstart
    positions = [x.copy()]
    velocities = [np.zeros(3)]
    travelled = 0.0

    #the walk, loop until the agent hits a wall or the max length is reached
    while travelled < max_len_m - 1e-12:
        #How much room is left to walk
        if travelled + speed > max_len_m + 1e-9:
            return None
        #add random noise to the heading
        heading = float(heading) + float(rng.normal(0.0, step_turn_std))
        #turns each heading into a 3D step
        step = speed * np.array([np.cos(heading), np.sin(heading)]) @ E.T
        #new position is the current position plus this step
        cand = x + step
        #check if the new position is outside the box
        outside = bool(np.any(np.abs(cand) > limit + 1e-9))
        #if the new position is inside the box, add the steps to the path
        if not outside:
            #take a step and record everything
            positions.append(cand)
            velocities.append(step)
            x = cand
            travelled += speed
            continue
        #shortens that step so it stops on the first wall it crosses
        point = _land_on_face(x, step, limit)
        if point is None:
            return None
        
        #if the step is outside the box, record the landing point
        velocities.append(point - x)
        positions.append(point)
        return (
            np.asarray(positions, dtype=float),
            np.asarray(velocities, dtype=float),
        )
    return None


def make_route(P, wall_in, cfg, w, rng, max_len_m=20.0):
    """One stored flight from P to the first wall and store it"""
    #Get the home perch
    P = np.asarray(P, dtype=float)
    #Get the limit of the box
    limit = float(w.limit)
    #Get the unit normal of the plane
    n = _normal_at(
        rng.uniform(cfg.tilt_min_deg, cfg.tilt_max_deg),
        rng.uniform(0.0, 2.0 * np.pi),
    )
    try:
        #pick an starting angle in the plane
        heading = takeoff_heading(n, wall_in, rng, cfg.takeoff_spread_deg)
    #plane is parallel to the perch wall error
    except RuntimeError:
        return None
    
    #Walks from the perch P on the plane until it hits a wall
    flown = fly_until_wall(
        P, n, heading, w.world_speed, w.step_turn_std, limit, rng,
        max_len_m=max_len_m)
    
    #If the walk is not possible - never landed on a wall over 20 m without a wall.
    if flown is None:
        return None
    #Store the path
    positions, velocities = flown
    #Get the last position
    Q = positions[-1].copy()
    if float(np.linalg.norm(Q - P)) < 1e-3:
        return None
    return dict(
        n_true=n.copy(),
        positions=positions,
        velocities=velocities,
        Q=Q,
    )


def build_routes(cfg, w, rng, max_draws=None):
    """Draw until rotes until we have enough routes."""
    P = np.asarray(HOME_PERCH, dtype=float)
    # Into the box from the perch on the left wall, (-1, 0, 0).
    wall_in = np.array([1.0, 0.0, 0.0])
    n_routes = int(cfg.n_routes)
    if max_draws is None:
        max_draws = 20 * n_routes
    max_draws = int(max_draws)
    #List of stored routes
    kept = []
    #Number of routes drawn
    drawn = 0
    #draw routes until we have enough routes.
    while len(kept) < n_routes:
        if drawn >= max_draws:
            raise RuntimeError(
                f"stopped after {drawn} draws with {len(kept)} of {n_routes} routes kept"
            )
        drawn += 1
        route = make_route(P, wall_in, cfg, w, rng)
        if route is not None:
            kept.append(route)
    return kept, drawn


def return_path(route):
    """The stored path reversed. Velocities are the position differences."""
    pos = np.asarray(route["positions"], dtype=float)[::-1].copy()
    vel = np.zeros_like(pos)
    vel[1:] = np.diff(pos, axis=0)
    return pos, vel


def build_schedule(n_routes, round_trips, rng):
    """A shuffle of each route repeated times."""
    seq = np.repeat(np.arange(int(n_routes)), int(round_trips))
    rng.shuffle(seq)
    return seq

def _blank():
    #Create the container for the blocks
    return {key: [] for key in ("pos", "vel", "n", "seg", "route", "flight", "plane")}


def _push(blocks, pos, vel, normal, segment, route_label, flight_label, plane):
    #add one chunck
    n_rows = len(pos)
    blocks["pos"].append(np.asarray(pos, dtype=float))
    blocks["vel"].append(np.asarray(vel, dtype=float))
    blocks["n"].append(np.repeat(np.asarray(normal, dtype=float)[None, :], n_rows, axis=0))
    blocks["seg"].append(np.full(n_rows, int(segment), dtype=int))
    blocks["route"].append(np.full(n_rows, int(route_label), dtype=int))
    blocks["flight"].append(np.full(n_rows, int(flight_label), dtype=int))
    blocks["plane"].append(np.full(n_rows, int(plane), dtype=int))


def _rest(blocks, where, n_steps, normal, segment, plane):
    #append a pause to the blocks above, when one round trip is done
    if int(n_steps) <= 0:
        return
    where = np.asarray(where, dtype=float)
    _push(
        blocks,
        np.repeat(where[None, :], int(n_steps), axis=0),
        np.zeros((int(n_steps), 3)),
        normal, segment, -1, -1, plane,
    )


def _route_table(routes):
    """One outbound row and one return row for each stored flight."""
    rows = []
    for r, route in enumerate(routes):
        n_true = np.asarray(route["n_true"], dtype=float).copy()
        rows.append({"route": 2 * r, "n_true": n_true.copy()})
        rows.append({"route": 2 * r + 1, "n_true": n_true.copy()})
    return rows


def make_planar_routes_trajectory(cfg, dt, scale, dim, rng=None, turn_std=None):
    """Build the trajectory from all the stored routes."""
    #turns dt and scale into the step length, the turn size, and the box half-width
    w = walk_params(cfg, dt, turn_std=turn_std, scale=scale)
    if rng is None:
        rng = w.rng
    
    #build the routes and the schedule
    routes, n_drawn = build_routes(cfg, w, rng)
    schedule = build_schedule(cfg.n_routes, cfg.round_trips, rng)
    
    #Get the home perch
    P = np.asarray(HOME_PERCH, dtype=float)
    #how long the rest are.
    rest_n = int(cfg.rest_steps)
    
    #Create the container for the blocks, and define parametes
    blocks = _blank()
    starts = []
    flight_table = []
    seen = np.zeros(int(cfg.n_routes), dtype=int)
    flight = 0
    t = 0

    #loop through the schedule, and build the blocks
    for trip, base in enumerate(schedule):
        #pick the stored flight for that round trip
        base = int(base)
        route = routes[base]
        #how many times this flight has been used already
        repeat = int(seen[base])
        seen[base] += 1
        out_label = 2 * base
        back_label = 2 * base + 1
        sheet = 2 * base
        normal = route["n_true"]

        #records the sample time index where this round trip begins
        starts.append(t)
        t0 = int(t)
        #Outbound flight, the stored path from the perch to the wall
        _push(
            blocks,
            route["positions"],
            route["velocities"],
            normal, trip, out_label, flight, sheet,
        )
        t += len(route["positions"])
        flight_table.append(dict(
            flight=int(flight), route=int(out_label), base_route=base,
            direction="out", t0=int(t0), t1=int(t),
        ))
        #Rest at the landing
        _rest(blocks, route["Q"], rest_n, normal, trip, sheet)
        t += rest_n
        flight += 1

        #Return flight, the stored path reversed.
        back_pos, back_vel = return_path(route)
        t0 = int(t)
        _push(
            blocks, back_pos, back_vel, normal,
            trip, back_label, flight, sheet,
        )
        t += len(back_pos)
        flight_table.append(dict(
            flight=int(flight), route=int(back_label), base_route=base,
            direction="back", t0=int(t0), t1=int(t),
        ))
        #Rest back at the perch.
        _rest(blocks, P, rest_n, normal, trip, sheet)
        t += rest_n
        flight += 1

    #Concatenate the blocks into the trajectory
    world = np.concatenate(blocks["pos"], axis=0)
    vel = np.concatenate(blocks["vel"], axis=0)
    n_true = np.concatenate(blocks["n"], axis=0)
    flight_of = np.concatenate(blocks["flight"], axis=0)
    plane_ids = np.concatenate(blocks["plane"], axis=0)
    torus_gt = torus_gt_from_velocity(vel, n_true, scale, dim)
    return Trajectory(
        world_pos=world,
        v_body_seq=vel,
        torus_gt=torus_gt,
        n_true_seq=n_true,
        segment_starts=np.asarray(starts, dtype=int),
        plane_id=plane_ids,
        flight_of=flight_of,
        route_table=_route_table(routes),
        flight_table=flight_table,
        n_route_draws=int(n_drawn),
    )


def columnar_ground_truth(traj, scale, dim):
    """Same walk, driven with the vertical normal. Ground truth is the horizontal frame."""
    n = np.tile(np.array([0.0, 0.0, 1.0]), (len(traj.world_pos), 1))
    gt = torus_gt_from_velocity(traj.v_body_seq, n, scale, dim)
    return n, gt


@dataclass
class Arena3DPlanarRoutesConfig(ExperimentConfig):
    """Round trips of stored flights from one perch."""
    n_routes: int = 40
    round_trips: int = 10
    takeoff_spread_deg: float = 60.0
    rest_steps: int = 500
    tilt_min_deg: float = 25.0
    tilt_max_deg: float = 50.0
    ratemap_n_sub: int = 600
    ratemap_n_shuffle: int = 0


class Arena3DPlanarRoutesExperiment(BaseExperiment):
    condition_label = "planar_routes"
    ratemap_ndim = 3

    def generate_trajectory(self, turn_std: float = None, n_steps=None, seed=None):
        rng = None if seed is None else np.random.default_rng(int(seed))
        return make_planar_routes_trajectory(
            self.config.experiment,
            dt=self.config.network.dt,
            scale=self.config.scale,
            dim=self.qan.manifold.dim,
            rng=rng,
            turn_std=turn_std,
        )
