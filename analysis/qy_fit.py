"""Qi and Yartsev single-cell lattice fit"""
import os
import time
import numpy as np
import torch
from scipy import sparse

from model.metrics import wrapped_angle_diff
from model.path_integration import frame

FLOOR = 0.01
SIGMA_FRAC = 0.125
N_PHASE = 14
N_SHUFFLES = 200
N_TARGET = 500
ORIENT_BATCH = 4 # A few orientations batches at once
SPACING_BATCH = 4 # A few spacings share one multiply


def plane_chart(world_pos, n_fit=None, x_home=None, basis=None):
    """Turn 3-D positions into 2-D coordinates in the chart plane.

    frame(n_fit) lays the chart normal flat. That is the generated movement plane of a
    planar route. A volumetric flight passes the SVD axes instead.
    """
    world_pos = np.asarray(world_pos, dtype=float)
    if x_home is None:
        x_home = world_pos[0]
    if basis is None:
        if n_fit is None:
            raise ValueError("plane_chart needs a basis or a chart normal")
        basis = frame(n_fit)
    else:
        basis = np.asarray(basis, dtype=float)
    return (world_pos - np.asarray(x_home, dtype=float)) @ basis


def align_field_trace(series, stride, n_samp):
    """Per-step labels at the same samples as the field trace (shortened recoding)."""
    out = np.asarray(series)[::int(stride)]
    return out[:int(n_samp)]


def spacing_values(n=75, s_min=0.24, s_max=1.0):
    """draw spacings valuesn each about 2% larger."""
    n = int(n)
    if n < 2:
        return np.array([float(s_min)])
    return float(s_min) * (float(s_max) / float(s_min)) ** (np.arange(n) / (n - 1))


def orientation_values():
    """0° through 59°. A 60° turn is the same hexagonal pattern."""
    return np.arange(60, dtype=float)


def hex_basis(spacing, orientation_deg):
    """Turns the spacing and orientation into a basis matrix."""
    spacing = float(spacing)
    theta = np.deg2rad(float(orientation_deg))
    
    #rotation is the matrix that turns a vector by theta degrees
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    raw = np.array([
        [spacing, 0.0],
        [0.5 * spacing, 0.5 * np.sqrt(3.0) * spacing],
    ])
    return raw @ rotation.T


def _lattice_g(xy, spacing, orientations, n_phase=N_PHASE, floor=FLOOR):
    """
    Build the firing-rate pattern of one hexagonal candidate and evaluate it at every position on the route,
    once for every phase. ``orientations`` is every angle in this group.
    """
    #basis matrix for the hexagonal candidate
    bases = np.stack(
        [hex_basis(spacing, float(angle)) for angle in orientations], axis=0,
    )
    basis = torch.from_numpy(np.ascontiguousarray(bases, dtype=np.float32)).to(xy.device)
    #width of the Gaussian
    sigma2 = (SIGMA_FRAC * float(spacing)) ** 2
    #rewrites every position in units of the basis
    coords = xy @ torch.linalg.inv(basis)

    #fractions of one step along one vectir
    frac = torch.arange(int(n_phase), device=xy.device, dtype=xy.dtype) / float(n_phase)
    #every slide along the first vecot
    pi, pj = torch.meshgrid(frac, frac, indexing="ij")
    #one row per phase
    phase = torch.stack((pi.reshape(-1), pj.reshape(-1)), dim=1)
    #subtract every phase from every position
    shifted = coords[:, None, :, :] - phase[None, :, None, :]
    #whole steps along each arrow: the lower corner of the cell
    u0 = torch.floor(shifted[..., 0])
    v0 = torch.floor(shifted[..., 1])
    #fractional parts add to more than 1 in the upper triangle of the cell
    upper = (shifted[..., 0] - u0) + (shifted[..., 1] - v0) > 1.0

    #finds the three lattice fields around each point and adds their Gaussians
    corners = (
        (torch.where(upper, u0 + 1.0, u0), v0),
        (torch.where(upper, u0, u0 + 1.0), torch.where(upper, v0 + 1.0, v0)),
        (torch.where(upper, u0 + 1.0, u0), v0 + 1.0),
    )

    total = torch.zeros(shifted.shape[:3], device=xy.device, dtype=xy.dtype)
    #visits the three corners
    for cu, cv in corners:
        delta = torch.stack((shifted[..., 0] - cu, shifted[..., 1] - cv), dim=-1) # how far the animal is from that corner, still in step units.
        world = delta @ basis[:, None, :, :] #turns that offset into a distance in the plane.
        dist2 = (world * world).sum(dim=-1)
        total = total + torch.exp(-dist2 / (2.0 * sigma2)) #adds a Gaussian of peak 1 and width
    return total + float(floor)


def _scoring_device():
    """GPU when the machine has one."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_sparse(matrix, device):
    """SciPy count table as a torch sparse matrix on ``device``."""
    torch.sparse.check_sparse_tensor_invariants.disable()
    coo = matrix.tocoo()
    indices = np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
    values = np.asarray(coo.data, dtype=np.float32)
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.ascontiguousarray(indices)),
        torch.from_numpy(np.ascontiguousarray(values)),
        size=(int(matrix.shape[0]), int(matrix.shape[1])),
    ).coalesce().to(device)


def log_likelihood_from_parts(n_spikes, occupancy, spike_term):
    """LL poisson log-likelihood"""
    return torch.special.xlogy(n_spikes, n_spikes / occupancy) + spike_term - n_spikes


def draw_spikes(activity, n_target, rng):
    """turns each cell's activity trace into integer spike counts. Using a poisson distribution."""
    activity = np.asarray(activity, dtype=float)
    if activity.ndim == 1:
        activity = activity[:, None]
    if np.any(activity < 0.0):
        raise ValueError("activity must be non-negative")
    totals = activity.sum(axis=0)
    scale = np.zeros(activity.shape[1], dtype=float)
    positive = totals > 0.0
    scale[positive] = float(n_target) / totals[positive]
    return rng.poisson(activity * scale[None, :]).astype(int)


def _lag_matrix(slices, n_shuffles, rng):
    """draws the time shifts used to shuffle spikes inside each flight. """
    lags = np.zeros((int(n_shuffles), len(slices)), dtype=int)
    #walk thorugh flights
    for j, (start, stop) in enumerate(slices):
        n = int(stop - start) #number of samples in the flight
        
        #10% and 90% of the flight
        lo = max(1, int(np.ceil(0.10 * n)))
        hi = int(np.floor(0.90 * n))
        if hi < lo:
            raise ValueError(
                f"a flight of {n} samples has no lag between 10% and 90%"
            )
        #Draws n_shuffles integers uniformly from lo through hi and stores
        lags[:, j] = rng.integers(lo, hi + 1, size=int(n_shuffles))
    return lags


def _routes(flight_of, flight_table, route_table):
    """One packed sequence per route.

    Qi and Yartsev fit each flight cluster on its own, and they find these
    clusters by grouping flights that follow nearly the same path from start
    to finish. A return starts at the other end, so it gets a route of its own.
    """
    normals = {}
    bases = {}
    for row in route_table:
        route_id = int(row["route"])
        normals[route_id] = np.asarray(row["n_fit"], dtype=float)
        if row.get("basis") is not None:
            bases[route_id] = np.asarray(row["basis"], dtype=float)

    #put all the flights for each route together
    grouped = {}
    for row in flight_table:
        grouped.setdefault(int(row["route"]), []).append(row)
     
    routes = []
    flight_of = np.asarray(flight_of)
    #For one flight, find every trace row with that flight's id   
    for route_id, rows in grouped.items():
        parts = []
        #returns every trace row whose label equals that flight's id
        for row in rows:
            idx = np.flatnonzero(flight_of == int(row["flight"]))
            if len(idx) < 2:
                raise ValueError(
                    f"flight {int(row['flight'])} has {len(idx)} trace samples; "
                    "a circular shift needs at least 2"
                )
            parts.append(np.asarray(idx, dtype=int))
        slices = []
        cursor = 0
        
        #Lays the flights end to end and records each oness place in that packed lis
        for part in parts:
            slices.append((cursor, cursor + len(part)))
            cursor += len(part) #start of the next flight. 
        routes.append(dict(
            route=int(route_id),
            index=np.concatenate(parts),
            slices=slices,
            n_fit=normals[route_id],
            basis=bases.get(route_id),
        ))
    return routes


def _spike_index(counts):
    """Sample time of every spike"""
    n_cells = counts.shape[0]
    lengths = counts.sum(axis=1).astype(int)
    width = int(lengths.max()) if n_cells else 0
    idx = np.zeros((n_cells, width), dtype=int)
    #marks valid cells
    valid = np.zeros((n_cells, width), dtype=bool)
    if width == 0:
        return idx, valid
    samples = np.arange(counts.shape[1])
    
    #fills one cell at a time.
    for c in range(n_cells):
        #no cells fired, stays 0
        if lengths[c] == 0:
            continue
        chosen = np.repeat(samples, counts[c])
        idx[c, :lengths[c]] = chosen
        valid[c, :lengths[c]] = True
    return idx, valid


def _shift_index(idx, valid, slices, lag_rows):
    """ Create one shuffled copy of the spike index for each shuffle."""
    n_shifts = len(lag_rows)
    shifted = np.broadcast_to(idx, (n_shifts,) + idx.shape).copy()
    for flight, (start, stop) in enumerate(slices):
        n = int(stop - start)
        lag = lag_rows[:, flight].astype(int)
        inside = valid[None, :, :] & (shifted >= start) & (shifted < stop)
        moved = start + np.mod(shifted - start + lag[:, None, None], n)
        shifted = np.where(inside, moved, shifted)
    return shifted



def _count_matrix(idx, valid, n_samples):
    """Spike counts as a table, with one row per spike train, one column per sample.
    """
    n_rows = int(np.prod(idx.shape[:-1]))  #one row per cell, or per shuffle x cell
    valid = np.broadcast_to(valid, idx.shape).reshape(n_rows, idx.shape[-1])
    idx = idx.reshape(n_rows, idx.shape[-1])
    rows = np.repeat(np.arange(n_rows), valid.sum(axis=1))  #row of every spike
    cols = idx[valid]  #sample of every spike
    #two spikes in the same row and sample are added up (count 2)
    return sparse.csr_matrix((np.ones(len(cols)), (rows, cols)),
                             shape=(n_rows, int(n_samples)))



def _z_against(value, null):
    """Z-score the LL scores against every row of the shuffles."""
    null = np.asarray(null, dtype=float)
    value = np.asarray(value, dtype=float)
    sd = null.std(axis=0, ddof=1) if len(null) > 1 else np.zeros_like(value)
    z = np.zeros(np.broadcast(value, sd).shape, dtype=float)
    ok = sd >= 1e-12 #avoid dividing by zero
    mean = null.mean(axis=0)
    #where ok is true, z becomes (LL score − shuffle mean) / shuffle spread.
    z[ok] = (np.broadcast_to(value, z.shape)[ok] - np.broadcast_to(mean, z.shape)[ok]) / sd[ok]
    return z


def _loo_z(values, chunk=256):
    """Gives each shuffle its own z-score using the others in comparison.

    The cell axis is scored in chunks. Each cell is independent, so the
    values match a single pass over the whole array.
    """
    values = np.asarray(values, dtype=float)
    n_cells = values.shape[-1]
    if n_cells <= int(chunk):
        return _loo_z_block(values)
    parts = [
        _loo_z_block(values[..., start:start + int(chunk)])
        for start in range(0, n_cells, int(chunk))
    ]
    return np.concatenate(parts, axis=-1)


def _loo_z_block(values):
    n = values.shape[0]
    if n < 3:
        return np.zeros_like(values)
    total = values.sum(axis=0)
    total_sq = np.sum(values * values, axis=0)
    #leaves out this row (values)
    loo_mean = (total - values) / (n - 1)
    loo_ss = (total_sq - values * values) - (n - 1) * loo_mean * loo_mean
    loo_sd = np.sqrt(np.maximum(loo_ss, 0.0) / (n - 2))
    z = np.zeros_like(values)
    ok = loo_sd >= 1e-12 #divide by zero
    z[ok] = (values[ok] - loo_mean[ok]) / loo_sd[ok]
    return z


def _mean_over_fired(z, fired):
    """Mean of Z values over routes a cell actually fired on``.
    """
    weight = np.asarray(fired, dtype=float)
    #how many routes the cell fired on.
    denom = weight.sum(axis=0)
    #z-scores added across routes
    total = (z * weight).sum(axis=-2)
    out = np.zeros(total.shape, dtype=float)
    active = denom > 0 #cells that fired on at least one route
    out[..., active] = total[..., active] / denom[active]
    return out


def _take(values, choice):
    """ pulls out the one spacing the fit chose."""
    choice = np.asarray(choice, dtype=int)
    pick = np.expand_dims(choice, axis=-2)
    return np.take_along_axis(values, pick, axis=-2)[..., 0, :]


def _streams(seed):
    """Independent spike and shift generators"""
    if seed is None:
        raise ValueError("seed is required and must differ per run")
    spike_seed, shift_seed = np.random.SeedSequence(int(seed)).spawn(2)
    return np.random.default_rng(spike_seed), np.random.default_rng(shift_seed)


def _default_jobs():
    return max(1, (os.cpu_count() or 1) - 1)


def _prepare_spikes(activity, seed, routes, n_target, n_shuffles):
    """Draw spikes and lags for this activity. Geometry (the chart) is shared."""
    spike_rng, shift_rng = _streams(seed)
    #concatenate all the activity traces for each route
    pieces = [activity[route["index"]] for route in routes]
    #draws spikes for each cell based on its activity trace
    spikes = draw_spikes(np.concatenate(pieces, axis=0), n_target, spike_rng)
    prepared = []
    cursor = 0
    #Walk the routes and their activity blocks together
    for route, block in zip(routes, pieces):
        #unpack the activity block and store for later in route
        n = len(block) #number of samples
        counts = spikes[cursor:cursor + n].T #Slice this route's spikes and transpose them to (cells, samples).
        spike_idx, spike_valid = _spike_index(counts)
        lags = _lag_matrix(route["slices"], n_shuffles, shift_rng)
        shifted_idx = _shift_index(spike_idx, spike_valid, route["slices"], lags)
        prepared.append(dict(
            n_spikes=counts.sum(axis=1).astype(int), #spike counts
            real=_count_matrix(spike_idx, spike_valid, n), #rows: cells
            shuffled=_count_matrix(shifted_idx, spike_valid, n), #rows: shuffle 0 cells, shuffle 1 cells, xxx
        ))
        cursor += n
    return prepared, spikes 


def _score_spacings(xy, spacings, orientations, n_phase, drawn, n_shuffles, counts=None):
    """Best orientation and phase for one route at each spacing.

    ``counts`` is the route's real and shuffled spike tables, already on the
    scoring device. Orientations are scored a few at a time. The spacings
    passed in share one multiply.
    """
    device = _scoring_device()
    n_cells = int(drawn["n_spikes"].shape[0])
    n_shuffles = int(n_shuffles)
    spacings = np.atleast_1d(np.asarray(spacings, dtype=float))
    n_spacings = len(spacings)
    if counts is None:
        real = _to_sparse(drawn["real"], device)
        shuf = _to_sparse(drawn["shuffled"], device)
    else:
        real, shuf = counts
    #routes positions in the plane
    xy = torch.from_numpy(np.ascontiguousarray(xy, dtype=np.float32)).to(device)
    n_spikes = torch.from_numpy(
        np.ascontiguousarray(drawn["n_spikes"], dtype=np.float32)
    ).to(device)

    best_real = torch.full((n_spacings, n_cells), -torch.inf, device=device)
    best_phase = torch.full((n_spacings, n_cells), -1, dtype=torch.long, device=device)
    best_orient = torch.full((n_spacings, n_cells), -1, dtype=torch.long, device=device)
    best_shuf = torch.full((n_spacings, n_shuffles, n_cells), -torch.inf, device=device)

    with torch.inference_mode():
        #Try orientations a few at a time.
        for start in range(0, len(orientations), ORIENT_BATCH):
            angles = orientations[start:start + ORIENT_BATCH]
            log_parts = []
            occ_parts = []
            #Build all phases for these orientations, at each spacing in the chunk.
            for spacing in spacings:
                g = _lattice_g(xy, float(spacing), angles, n_phase=n_phase)
                n_phase_grid = g.shape[1]
                n_orient = g.shape[0]
                #log g turned to one row per sample, one column per orientation and phase
                log_parts.append(
                    torch.log(g).permute(2, 0, 1).reshape(g.shape[2], -1))
                #Sum the lattice fields at every point to get occupancy.
                occ_parts.append(g.sum(dim=-1))
            log_g = torch.cat(log_parts, dim=1).contiguous()
            occupancy = torch.stack(occ_parts, dim=0)
            #Find the best phase, then the best orientation in this group.
            real_term = torch.sparse.mm(real, log_g).reshape(
                n_cells, n_spacings, n_orient, n_phase_grid)
            ll = log_likelihood_from_parts(
                n_spikes[:, None, None, None], occupancy[None], real_term)
            ll_phase, phase_i = torch.max(ll, dim=-1)
            ll_o, o_local = torch.max(ll_phase, dim=-1)
            ll_o = ll_o.transpose(0, 1)
            o_local = o_local.transpose(0, 1)
            # better is true for each cell whose likelihood beat the best orientation so far
            better = ll_o > best_real
            chosen_phase = torch.gather(
                phase_i, 2, o_local.transpose(0, 1).unsqueeze(-1),
            ).squeeze(-1).transpose(0, 1)
            best_real = torch.where(better, ll_o, best_real)
            best_phase = torch.where(better, chosen_phase, best_phase)
            best_orient = torch.where(better, o_local + start, best_orient)

            #Calculate LL for each shuffle, best phase and orientation.
            terms = torch.sparse.mm(shuf, log_g).reshape(
                n_shuffles, n_cells, n_spacings, n_orient, n_phase_grid)
            ll_s = log_likelihood_from_parts(
                n_spikes[None, :, None, None, None], occupancy[None, None], terms)
            best_shuf = torch.maximum(
                best_shuf, ll_s.amax(dim=(-1, -2)).permute(2, 0, 1))

    return (
        best_real.cpu().numpy(),
        best_shuf.cpu().numpy(),
        best_orient.cpu().numpy().astype(int),
        best_phase.cpu().numpy().astype(int),
    )


def _score_pair(xy, spacing, orientations, n_phase, drawn, n_shuffles, counts=None):
    """Best orientation and phase for one route at one spacing."""
    real, shuf, orient, phase = _score_spacings(
        xy, [spacing], orientations, n_phase, drawn, n_shuffles, counts)
    return real[0], shuf[0], orient[0], phase[0]


def _pair_task(r_i, s_i, xy, spacing, orientations, n_phase, drawn, n_shuffles, counts=None):
    return r_i, s_i, _score_pair(
        xy, spacing, orientations, n_phase, drawn, n_shuffles, counts)


def _weighted_mean(total, fired):
    """Mean over routes. ``fired`` weights each route; a silent route adds nothing."""
    denom = np.asarray(fired, dtype=float).sum(axis=0)
    out = np.zeros_like(total, dtype=float)
    active = denom > 0.0
    out[..., active] = total[..., active] / denom[active]
    return out


def _assemble(routes, spacings, orientations, n_target, n_shuffles, n_cells,
              real_ll, z_route, shuf_totals, mean_loo,
              real_orient, real_phase, fired, spikes, return_spikes):
    #Turns the per-route log-likelihood tables into one spacing choice and one grid-fit score per cell
    if z_route.shape != real_ll.shape:
        raise RuntimeError(
            f"real z-score has shape {z_route.shape}, expected {real_ll.shape}"
        )
    #pick one spacing
    mean_z = _mean_over_fired(z_route, fired)
    spacing_choice = np.argmax(mean_z, axis=0)
    #score the data on this scoring
    real_totals = real_ll.sum(axis=1)
    cell_total = real_totals[spacing_choice, np.arange(n_cells)]
    null_at_choice = _take(
        shuf_totals, np.broadcast_to(spacing_choice, (int(n_shuffles), n_cells)),
    )
    grid_fit = _z_against(cell_total, null_at_choice)
    #Build a null that also gets to pick its spacing
    choice_j = np.argmax(mean_loo, axis=1)
    z_loo_total = _loo_z(shuf_totals)
    null_scores = _take(z_loo_total, choice_j)

    #Build the output, One record per cell
    cells = []
    active = []
    null_medians = []
    for c in range(n_cells):
        n_spikes = int(sum(int(route["n_spikes"][c]) for route in routes))
        silent = n_spikes == 0
        #The 95th percentile of the null scores
        threshold = float(np.percentile(null_scores[:, c], 95))
        cells.append(dict(
            grid_fit=float("nan") if silent else float(grid_fit[c]),
            significant=False if silent else bool(grid_fit[c] > threshold),
            threshold=threshold,
            spacing=float(spacings[int(spacing_choice[c])]),
            spacing_index=int(spacing_choice[c]),
            orientation=real_orient[int(spacing_choice[c]), :, c].astype(int).copy(),
            phase=real_phase[int(spacing_choice[c]), :, c].astype(int).copy(),
            n_spikes=n_spikes,
            null_median=float(np.median(null_scores[:, c])),
        ))
        if not silent:
            active.append(cells[-1])
            null_medians.append(cells[-1]["null_median"])
    out = dict(
        cells=cells,
        routes=[int(route["route"]) for route in routes],
        spacings=spacings,
        orientations=orientations,
        n_target=int(n_target),
        n_silent=int(n_cells - len(active)),
        median_grid_fit=(
            float(np.median([cell["grid_fit"] for cell in active]))
            if active else float("nan")
        ),
        median_null_score=(
            float(np.median(null_medians)) if null_medians else float("nan")
        ),
        frac_significant=(
            float(np.mean([cell["significant"] for cell in active]))
            if active else float("nan")
        ),
    )
    if return_spikes:
        out["spikes"] = spikes
    return out


def fit_routes(positions, activity, flight_of, flight_table, route_table, *,
               seed, n_target=N_TARGET, n_shuffles=N_SHUFFLES,
               spacings=None, orientations=None, n_phase=N_PHASE,
               return_spikes=False, n_jobs=None):
    """
    Fits one spacing per cell. It picks one spacing for the cell
    and lets orientation and phase be chosen separately on every route.
    """
    # Time the prepare
    t0 = time.perf_counter()
    #setup everything
    positions = np.asarray(positions, dtype=float)
    activity = np.asarray(activity, dtype=float)
    routes = _routes(flight_of, flight_table, route_table)
    if not routes:
        raise ValueError("no flights to fit")
    if spacings is None:
        spacings = spacing_values()
    if orientations is None:
        orientations = orientation_values()
    spacings = np.asarray(spacings, dtype=float)
    orientations = np.asarray(orientations, dtype=float)
    for route in routes:
        route["xy"] = plane_chart(
            positions[route["index"]], route["n_fit"], basis=route["basis"],
        )
    drawn, spikes = _prepare_spikes(activity, seed, routes, n_target, n_shuffles)  # CHANGED: spikes returned separately
    # Shuffle scores are reduced one route at a time. The full
    # (shuffle, spacing, route, cell) tensor is not stored.
    n_cells = activity.shape[1]
    n_routes = len(routes)
    n_spacings = len(spacings)
    n_shuf = int(n_shuffles)
    real_ll = np.full((n_spacings, n_routes, n_cells), -np.inf)
    z_route = np.zeros((n_spacings, n_routes, n_cells))
    shuf_totals = np.zeros((n_shuf, n_spacings, n_cells))
    loo_weighted = np.zeros((n_shuf, n_spacings, n_cells))
    real_orient = np.full((n_spacings, n_routes, n_cells), -1, dtype=int)
    real_phase = np.full((n_spacings, n_routes, n_cells), -1, dtype=int)
    fired = np.stack([item["n_spikes"] for item in drawn], axis=0) > 0

    device = _scoring_device()
    workers = _default_jobs() if n_jobs is None else int(n_jobs)
    # One GPU: spacings stay in this process and reuse the uploaded counts.
    if device.type != "cpu":
        workers = 1
    parallel = None
    if workers > 1:
        from joblib import Parallel, delayed
        parallel = Parallel(n_jobs=workers)
    # Time the prepare
    t1 = time.perf_counter()
    print(f"qy prepare {t1 - t0:.2f}s")

    #Search each route for the best spacing, orientation, and phase.
    for r_i, route in enumerate(routes):
        counts = None
        if device.type != "cpu":
            counts = (
                _to_sparse(drawn[r_i]["real"], device),
                _to_sparse(drawn[r_i]["shuffled"], device),
            )
        if parallel is None:
            #Score each spacing in chunks
            for s0 in range(0, n_spacings, SPACING_BATCH):
                chunk = spacings[s0:s0 + SPACING_BATCH]
                best_real, best_shuf, best_orient, best_phase = _score_spacings(
                    route["xy"], chunk, orientations, n_phase,
                    drawn[r_i], n_shuffles, counts)
                
                #writes one spacing's scores into the tables for the whole search.
                for j, s_i in enumerate(range(s0, s0 + len(chunk))):
                    real_ll[s_i, r_i] = best_real[j]
                    real_orient[s_i, r_i] = best_orient[j]
                    real_phase[s_i, r_i] = best_phase[j]
                    z_route[s_i, r_i] = _z_against(best_real[j], best_shuf[j])
                    shuf_totals[:, s_i] += best_shuf[j]
                    weight = fired[r_i].astype(float)
                    loo_weighted[:, s_i] += _loo_z(best_shuf[j]) * weight
            continue
        pairs = [
            (r_i, s_i, route["xy"], float(spacing), orientations, n_phase,
             drawn[r_i], n_shuffles, counts)
            for s_i, spacing in enumerate(spacings)
        ]
        scored = parallel(delayed(_pair_task)(*item) for item in pairs)

        for _, s_i, (best_real, best_shuf, best_orient, best_phase) in scored:
            real_ll[s_i, r_i] = best_real
            real_orient[s_i, r_i] = best_orient
            real_phase[s_i, r_i] = best_phase
            z_route[s_i, r_i] = _z_against(best_real, best_shuf)
            shuf_totals[:, s_i] += best_shuf
            weight = fired[r_i].astype(float)
            loo_weighted[:, s_i] += _loo_z(best_shuf) * weight
    # Time the search
    t2 = time.perf_counter()
    print(f"qy search {t2 - t1:.2f}s")

    for route, item in zip(routes, drawn):
        route["n_spikes"] = item["n_spikes"]
    mean_loo = _weighted_mean(loo_weighted, fired)
    out = _assemble(
        routes, spacings, orientations, n_target, n_shuffles, n_cells,
        real_ll, z_route, shuf_totals, mean_loo,
        real_orient, real_phase,
        fired, spikes, return_spikes,  # CHANGED: was drawn[0]["spikes"]
    )
    # Time the assemble
    t3 = time.perf_counter()
    print(f"qy assemble {t3 - t2:.2f}s")
    return out


def qy_fit(traj, field_pos, field_act, stride, **kwargs):
    """Align the fligh ids (shortening them) to the field trace (shortened recoding), then fit."""
    n_samp = len(field_pos)
    flight_of = align_field_trace(traj.flight_of, stride, n_samp)
    return fit_routes(
        field_pos, field_act, flight_of, traj.flight_table, traj.route_table,
        **kwargs,
    )


def repeat_phase_misalignment(theta, flight_table):
    """Phase difference between repeats of one stored path, in radians.

    Measures how far the network's internal phase drifts when the same
    path is flown again.
    """
    theta = np.asarray(theta, dtype=float) #internal phase of the network
    grouped = {}
    #loop thorugh flight table when they start and end
    for row in flight_table:
        grouped.setdefault(int(row["route"]), []).append(row)

    rows = []
    #walks thorugh stored paths. Compare first pass with later passes.
    for route_id, flights in grouped.items():
        flights = sorted(flights, key=lambda item: int(item["t0"]))
        ref = theta[int(flights[0]["t0"]):int(flights[0]["t1"])]
        diffs = []

        #Later passes of the pass , skip the first one.
        for row in flights[1:]:
            cur = theta[int(row["t0"]):int(row["t1"])]
            n = min(len(ref), len(cur))
            diffs.append(wrapped_angle_diff(cur[:n], ref[:n]))
        if diffs:
            stacked = np.concatenate(diffs, axis=0)
            rms = float(np.sqrt(np.mean(np.square(stacked)))) #root-mean-square of the phase differences
        else:
            rms = 0.0
        rows.append(dict(route=int(route_id), n_repeats=len(flights), rms_rad=rms))
    return rows
