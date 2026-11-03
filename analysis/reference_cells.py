import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from config import world_to_flat_bins


def reference_activity(kind, n_cells, world_pos, spacing, field_sd, env_size, seed=0):
    """Genrates protopyes of lattices in the 3D box"""
    rng = np.random.default_rng(seed)
    S, half, margin = spacing, env_size / 2, 4 * field_sd
    
    #how many lattice steps to build in each direction
    m = int(np.ceil(2 * (np.sqrt(3) * half + 2 * S + margin) / S)) + 2 
    #The points in the lattice
    i, j, k = np.meshgrid(*[np.arange(-m, m + 1)] * 3, indexing="ij")
    
    #Build the repeating pattern depedning on the kind of lattice
    layer = k % (3 if kind == "fcc" else 2)  # ABCABC (FCC) or ABABAB (HCP)
    lattice = np.stack([(i + j / 2 + layer / 2) * S, #x coordinates
                        (j * np.sqrt(3) / 2 + layer / (2 * np.sqrt(3))) * S, #y coordinates
                        k * np.sqrt(2 / 3) * S], -1).reshape(-1, 3) #z coordinates
    #flat hexagons in the xy plane used for COL 
    columns = np.stack([(i + j / 2) * S, j * np.sqrt(3) / 2 * S], -1)[:, :, 0].reshape(-1, 2)
    
    #For each cell loop over and generate the activity rates put into output array
    out = np.empty((len(world_pos), n_cells), np.float32)
    for c in range(n_cells):
        #Every branch generates a tree (field centres) and query points (positions on the walk)
        
        if kind == "rnd":   #randomly distributed cells in the box
            n = int(round(np.sqrt(2) / S ** 3 * (2 * (half + margin)) ** 3))
            #random cells in three, whole trajectory in q
            tree, q = cKDTree(rng.uniform(-half - margin, half + margin, (n, 3))), world_pos #
        elif kind == "col":     # striped cell firing, columns
            a = rng.uniform(0, 2 * np.pi)
            #turned hexagonal floor pattern in tree, and walks x and Y in q
            turn = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            tree, q = cKDTree(columns @ turn.T + rng.uniform(-S, S, 2)), world_pos[:, :2]
        else: #fcc and hcp: the rotated crystal in tree, and the whole walk in q.
            turn = Rotation.random(random_state=int(rng.integers(0, 2**31)))
            p = turn.apply(lattice) + rng.uniform(-S, S, 3)
            tree, q = cKDTree(p[np.all(np.abs(p) <= half + margin, axis=1)]), world_pos
        d, _ = tree.query(q)
        out[:, c] = np.exp(-0.5 * (d / field_sd) ** 2)
    return out


def reference_sums(world_pos, act, env_size, bins):
    """Rate-map sums (b, b, b, n) and counts (b, b, b), with the network's own binning."""
    flat = world_to_flat_bins(world_pos, env_size, bins, ndim=3)
    counts = np.bincount(flat, minlength=bins ** 3).reshape((bins,) * 3).astype(float)
    sums = np.stack([np.bincount(flat, act[:, c], bins ** 3) for c in range(act.shape[1])], -1)
    return sums.reshape((bins,) * 3 + (-1,)), counts
