"""Qi and Yartsev single-cell lattice fit"""
import numpy as np
from scipy.special import xlogy

from model.metrics import field_spacing_rad, wrapped_angle_diff
from model.path_integration import frame

FLOOR = 0.01
SIGMA_FRAC = 0.125
N_PHASE = 14
N_SHUFFLES = 200
N_TARGET = 500


def plane_chart(world_pos, n_true, x_home=None):
    """Turn 3-D positions into 2-D coordinates in the plane of the true normal."""
    world_pos = np.asarray(world_pos, dtype=float)
    if x_home is None:
        x_home = world_pos[0]
    return (world_pos - np.asarray(x_home, dtype=float)) @ frame(n_true)


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


def lattice_g(xy, spacing, orientation_deg, n_phase=N_PHASE, floor=FLOOR):
    """
    Build the firing-rate pattern of one hexagonal candidate and evaluates it at every position on the route, 
    once for every phase.
    """
    #routes positions in the plane
    xy = np.asarray(xy, dtype=float)
    #basis matrix for the hexagonal candidate
    basis = hex_basis(spacing, orientation_deg)
    #width of the Gaussian
    sigma2 = (SIGMA_FRAC * float(spacing)) ** 2
    #rewrites every position in units of the basis
    coords = xy @ np.linalg.inv(basis)

    #fractions of one step along one vectir
    frac = np.arange(int(n_phase)) / float(n_phase)
    #every slide along the first vecot
    pi, pj = np.meshgrid(frac, frac, indexing="ij")
    #one row per phase
    phase = np.stack([pi.ravel(), pj.ravel()], axis=1)
    #subtract every phase from every position
    shifted = coords[None, :, :] - phase[:, None, :]
    #whole steps along each arrow: the lower corner of the cell
    u0 = np.floor(shifted[..., 0])
    v0 = np.floor(shifted[..., 1])
    #fractional parts add to more than 1 in the upper triangle of the cell
    upper = (shifted[..., 0] - u0) + (shifted[..., 1] - v0) > 1.0
    
    #finds the three lattice fields around each point and adds their Gaussians
    corners = (
        (np.where(upper, u0 + 1.0, u0), v0),
        (np.where(upper, u0, u0 + 1.0), np.where(upper, v0 + 1.0, v0)),
        (np.where(upper, u0 + 1.0, u0), v0 + 1.0),
    )
    
    total = np.zeros(shifted.shape[:2], dtype=float)
    #visits the three corners
    for cu, cv in corners:
        delta = np.stack([shifted[..., 0] - cu, shifted[..., 1] - cv], axis=-1) # how far the animal is from that corner, still in step units.
        world = delta @ basis #turns that offset into a distance in the plane.
        dist2 = np.sum(world * world, axis=-1)
        total += np.exp(-dist2 / (2.0 * sigma2)) #adds a Gaussian of peak 1 and width
    return total + float(floor)


def closed_form_amplitude(n_spikes, occupancy):
    """Gives the amplitude that maximises the log-likelihood."""
    occupancy = float(occupancy)
    if occupancy <= 0.0:
        raise ValueError("occupancy must be positive")
    return float(n_spikes) / occupancy


def log_likelihood_from_parts(n_spikes, occupancy, spike_term):
    """LL poisson log-likelihood"""
    n_spikes = np.asarray(n_spikes, dtype=float)
    occupancy = np.asarray(occupancy, dtype=float)
    spike_term = np.asarray(spike_term, dtype=float)
    return xlogy(n_spikes, n_spikes / occupancy) + spike_term - n_spikes


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


def _routes(flight_of, flight_table, route_table, qy_routes):
    "takes the downsampled recording and builds one packed sequence per route"
    if qy_routes not in ("out", "out_and_back"):
        raise ValueError(f"unknown qy_routes {qy_routes!r}")
    keep = {"out"} if qy_routes == "out" else {"out", "back"}
    normals = {int(row["route"]): np.asarray(row["n_true"], dtype=float) for row in route_table}
    
    #put all the flights for each route together
    grouped = {}
    for row in flight_table:
        direction = row.get("direction", "out")
        if direction not in keep:
            continue
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
            n_true=normals[route_id],
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


def _spike_term(log_g, idx, valid):
    """Adds up log g at every spike time."""
    n_phase = log_g.shape[0]
    if idx.shape[-1] == 0:
        return np.zeros(idx.shape[:-1] + (n_phase,), dtype=float)
    lead = idx.shape[:-1]
    flat_idx = np.where(valid, idx, 0).reshape(-1, idx.shape[-1])
    flat_valid = np.reshape(valid, (-1, idx.shape[-1]))
    gathered = log_g[:, flat_idx]
    gathered = np.where(flat_valid[None, :, :], gathered, 0.0)
    terms = gathered.sum(axis=-1)
    return np.moveaxis(terms, 0, -1).reshape(lead + (n_phase,))


def _best_phase(n_spikes, occupancy, spike_term):
    """Best log-likelihood over a phase"""
    ll = log_likelihood_from_parts(
        n_spikes[:, None], occupancy[None, :], spike_term)
    choice = np.argmax(ll, axis=1)
    return ll[np.arange(len(choice)), choice], choice


def _best_phase_shifts(n_spikes, occupancy, spike_term):
    """Best log-likelihood over phase for several shifts at once.
    """
    ll = log_likelihood_from_parts(
        n_spikes[None, :, None], occupancy[None, None, :], spike_term)
    choice = np.argmax(ll, axis=-1)
    return np.take_along_axis(ll, choice[..., None], axis=-1)[..., 0]


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


def _loo_z(values):
    """Gives each shuffle its own z-score using the others in comparison """
    values = np.asarray(values, dtype=float)
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
        raise ValueError("seed is required and must differ per run and arm")
    spike_seed, shift_seed = np.random.SeedSequence(int(seed)).spawn(2)
    return np.random.default_rng(spike_seed), np.random.default_rng(shift_seed)


def fit_routes(positions, activity, flight_of, flight_table, route_table, *,
               seed, n_target=N_TARGET, qy_routes="out", n_shuffles=N_SHUFFLES,
               spacings=None, orientations=None, n_phase=N_PHASE):
    """
    Fits one spacing per cell. It picks one spacing for the cell 
    and lets orientation and phase be chosen separately on every route.
    """
    #setup everything
    positions = np.asarray(positions, dtype=float)
    activity = np.asarray(activity, dtype=float)
    routes = _routes(flight_of, flight_table, route_table, qy_routes)
    if spacings is None:
        spacings = spacing_values()
    if orientations is None:
        orientations = orientation_values()
    spacings = np.asarray(spacings, dtype=float)
    orientations = np.asarray(orientations, dtype=float)
    spike_rng, shift_rng = _streams(seed)
    #concatenate all the activity traces for each route
    pieces = [activity[route["index"]] for route in routes]
    #draws spikes for each cell based on its activity trace
    spikes = draw_spikes(np.concatenate(pieces, axis=0), n_target, spike_rng)
    cursor = 0
    #Walk the routes and their activity blocks together
    for route, block in zip(routes, pieces):
        #unpack the activity block and store for later in route
        n = len(block) #number of samples
        counts = spikes[cursor:cursor + n].T #Slice this route's spikes and transpose them to (cells, samples).
        route["n_spikes"] = counts.sum(axis=1).astype(int) #spike counts
        route["spike_idx"], route["spike_valid"] = _spike_index(counts)
        route["xy"] = plane_chart(positions[route["index"]], route["n_true"])
        route["lags"] = _lag_matrix(route["slices"], n_shuffles, shift_rng)
        cursor += n

    #allocate empty scoring tables
    n_cells = activity.shape[1]
    n_routes = len(routes)
    n_spacings = len(spacings)
    real_ll = np.full((n_spacings, n_routes, n_cells), -np.inf)
    shuf_ll = np.full((int(n_shuffles), n_spacings, n_routes, n_cells), -np.inf)
    real_orient = np.full((n_spacings, n_routes, n_cells), -1, dtype=int)
    real_phase = np.full((n_spacings, n_routes, n_cells), -1, dtype=int)
    fired = np.stack([route["n_spikes"] for route in routes], axis=0) > 0

    chunk = 16 #number of shuffles scored at one time.
    #Search each route for the best spacing, orientation, and phase.
    for r_i, route in enumerate(routes):
        n_spikes = route["n_spikes"].astype(float) #spike counts
        shifted_idx = _shift_index(
            route["spike_idx"], route["spike_valid"], route["slices"], route["lags"])
        shifted_valid = np.broadcast_to(
            route["spike_valid"], shifted_idx.shape)
        
        #Try each spacing.
        for s_i, spacing in enumerate(spacings):
            best_real = np.full(n_cells, -np.inf)
            best_phase = np.full(n_cells, -1, dtype=int)
            best_orient = np.full(n_cells, -1, dtype=int)
            best_shuf = np.full((int(n_shuffles), n_cells), -np.inf)
            
            #Try each orientation.
            for o_i, orientation in enumerate(orientations):
                #Build all phases for this orientation.
                g = lattice_g(route["xy"], spacing, orientation, n_phase=n_phase)
                log_g = np.log(g)
                #Sum the lattice fields at every point to get occupancy.
                occupancy = g.sum(axis=1)
                #Find the best phase for this orientation
                ll, phase = _best_phase(
                    n_spikes,
                    occupancy,
                    _spike_term(log_g, route["spike_idx"], route["spike_valid"]),
                )
                # better is true for each cell whose likelihood beat the best orientation so far
                better = ll > best_real
                
                best_real[better] = ll[better]
                best_phase[better] = phase[better]
                best_orient[better] = o_i
                
                #Calculate LL for each shuffle.
                for start in range(0, int(n_shuffles), chunk):
                    stop = min(start + chunk, int(n_shuffles))
                    terms = _spike_term(
                        log_g, shifted_idx[start:stop], shifted_valid[start:stop])
                    
                    #Find the best phase for each shuffle.
                    ll_s = _best_phase_shifts(n_spikes, occupancy, terms)
                    block = best_shuf[start:stop]
                    # better is true for each cell whose likelihood beat the best orientation so far
                    better_s = ll_s > block
                    block[better_s] = ll_s[better_s]
            
            #copy the best that survived into next spacing
            real_ll[s_i, r_i] = best_real
            shuf_ll[:, s_i, r_i] = best_shuf
            real_orient[s_i, r_i] = best_orient
            real_phase[s_i, r_i] = best_phase

    #Turns the per-route log-likelihood tables into one spacing choice and one grid-fit score per cell
    z_route = _z_against(real_ll, shuf_ll)
    if z_route.shape != real_ll.shape:
        raise RuntimeError(
            f"real z-score has shape {z_route.shape}, expected {real_ll.shape}"
        )
    #pick one spacing
    mean_z = _mean_over_fired(z_route, fired)
    spacing_choice = np.argmax(mean_z, axis=0)
    
    #score the data on this scoring
    real_totals = real_ll.sum(axis=1)
    shuf_totals = shuf_ll.sum(axis=2)
    cell_total = real_totals[spacing_choice, np.arange(n_cells)]
    null_at_choice = _take(shuf_totals, np.broadcast_to(spacing_choice, (int(n_shuffles), n_cells)))
    grid_fit = _z_against(cell_total, null_at_choice)

    #Build a null that also gets to pick its spacing
    z_loo_route = _loo_z(shuf_ll)
    mean_loo = _mean_over_fired(z_loo_route, fired)
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
    return dict(
        cells=cells,
        spikes=spikes,
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


def qy_fit(traj, field_pos, field_act, stride, **kwargs):
    """Align the fligh ids (shortening them) to the field trace (shortened recoding), then fit."""
    n_samp = len(field_pos)
    flight_of = align_field_trace(traj.flight_of, stride, n_samp)
    return fit_routes(
        field_pos, field_act, flight_of, traj.flight_table, traj.route_table,
        **kwargs,
    )


def repeat_phase_misalignment(theta, flight_table, n_sheet, bump_spacing_cells):
    """Phase difference between repeats of one stored path, in grid periods.
     Measures how far the network's internal phase drifts when the same outbound 
     path is flown again.
    """
    theta = np.asarray(theta, dtype=float) #internal phase of the network
    period = field_spacing_rad(n_sheet, bump_spacing_cells)
    grouped = {}
    #loop thorugh flight table when they start and end
    for row in flight_table:
        if row.get("direction", "out") != "out": #only look at outbound flights
            continue
        grouped.setdefault(int(row.get("base_route", row["route"])), []).append(row)
    
    rows = []
    #walks thorugh stored paths. Compare first pass with later passes.
    for base, flights in grouped.items():
        flights = sorted(flights, key=lambda item: int(item["t0"]))
        ref = theta[int(flights[0]["t0"]):int(flights[0]["t1"])]
        diffs = []
        
        #Later passes of the pass , skip the first one.
        for row in flights[1:]:
            cur = theta[int(row["t0"]):int(row["t1"])]
            n = min(len(ref), len(cur))
            diffs.append(wrapped_angle_diff(cur[:n], ref[:n]) / period)
        if diffs:
            stacked = np.concatenate(diffs, axis=0)
            rms = float(np.sqrt(np.mean(np.square(stacked)))) #root-mean-square of the phase differences
        else:
            rms = 0.0
        rows.append(dict(base_route=int(base), n_repeats=len(flights), rms_periods=rms))
    return rows
