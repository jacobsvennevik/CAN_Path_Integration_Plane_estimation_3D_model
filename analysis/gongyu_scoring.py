"""
Gong & Yu's scoring logic, pasted almost verbatum to keep consistency.

"""

import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import correlate, find_peaks
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import MeanShift

CHI_AZIMUTHS_DEG  = np.arange(22, 360, 30)        # 12 azimuths actually sampled
CHI_ALTITUDES_DEG = (72, 56)                       # (hexagonal-plane, square-plane)
CHI_TEMPLATE = {                                   # (hgs azimuth-idx, sgs azimuth-idx)
    "fcc": ([3, 7, 11], [1, 5, 9]),
    "hcp": ([1, 5, 9], [3, 7, 11]),
    "col": ([0, 2, 4, 6, 8, 10], [0, 2, 4, 6, 8, 10]),
}
RADIAL_FRACTION = 0.3                              # ring search radius = int(d * frac)


def rot_x(x, azimuth, altitude):
    ''' 
    Rotates x depending on the different ways of lloking at the firing fields.
    Done by rotating the positions with the given azimuth and altitude as the positive direction
    '''
    cz, sz = np.cos(azimuth), np.sin(azimuth)
    cl, sl = np.cos(altitude), np.sin(altitude)
    rot1 = np.array([[cl, 0, -sl],
                     [0, 1, 0],
                     [sl, 0, cl]])
    rot2 = np.array([[cz, sz, 0],
                     [-sz, cz, 0],
                     [0, 0, 1]])
    return x @ (rot2 @ rot1).T


def oblique_slice(ac, azimuth, altitude, bins=51):
    # Stack coordinates
    x, y, z = ac.shape[:-1] if len(ac.shape) == 4 else ac.shape
    x, y, z = np.linspace(-1, 1, x), np.linspace(-1, 1, y), np.linspace(-1, 1, z)

    ac_interpolator = RegularGridInterpolator((x, y, z), ac, bounds_error=False)

    x0 = np.linspace(-1, 1, bins)
    plane0 = np.stack(np.meshgrid(x0, x0, indexing='ij'), axis=-1).reshape(-1, 2)
    plane0 = np.hstack((plane0, np.zeros((plane0.shape[0], 1))))

    plane = rot_x(plane0, azimuth=azimuth, altitude=altitude)

    return np.nan_to_num(ac_interpolator(plane)).reshape(bins, bins, -1)


def autocorr(f, th):
    ''' Standardized autocorrelation
    '''
    if len(f.shape) == 3:
        f = f[..., None]
    n = np.prod(f.shape[:-1])
    f_ = (f - f.mean(axis=(0, 1, 2))) / f.std(axis=(0, 1, 2))

    acs = []
    for i in range(f.shape[-1]):
        ac = correlate(f_[..., i], f_[..., i], mode='full') / n
        ac[ac < th] = 0
        acs.append(ac)
    return np.stack(acs, axis=-1)

def rate_map(x, aang, precision=25, reshape=False):
    ''' Generate 3d histogram for firing rates
    '''
    d = x.shape[-1]
    xrange = np.linspace(-1, 1, precision, endpoint=True)
    n, n_neurons = len(xrange) - 1, aang.shape[-1]
    r = np.zeros((n, n, n, n_neurons)) if d == 3 else np.zeros((n, n, n_neurons))
    for i0 in range(n):
        regionx = (x[:, 0] > xrange[i0]) & (x[:, 0] <= xrange[i0+1])
        for i1 in range(n):
            regiony = (x[:, 1] > xrange[i1]) & (x[:, 1] <= xrange[i1+1])
            regionxy = regionx & regiony
            if d == 3:
                for i2 in range(n):
                    regionz = (x[:, 2] > xrange[i2]) & (x[:, 2] <= xrange[i2+1])
                    region = regionxy & regionz
                    if np.any(region):
                        r[i0, i1, i2] = aang[region].mean(axis=0)
            elif np.any(region): # 2D
                r[i0, i1] = aang[region].mean(axis=0)

    xs = (xrange[1:] + xrange[:-1]) / 2
    xs = np.meshgrid(xs, xs, xs, indexing='ij') if d == 3 else np.meshgrid(xs, xs, indexing='ij')
    xs = np.stack(xs, axis=-1)
    if reshape:
        xs = xs.reshape(-1, d)
        r = r.reshape(-1, n_neurons)
    return r, xs


def hist3d(x, spikes=None, bins=30, lim=((-1, 1), (-1, 1), (-1, 1)), sigma=0):
    ''' A faster and more generic version of `rate_map`
    '''
    if spikes is None:
        f, xs = np.histogramdd(x, bins=bins, range=lim, density=False)
        f = f / f.sum()
    else:
        if len(spikes.shape) == 1:
            spikes = spikes[:, None]
        f = []
        for i in range(spikes.shape[-1]):
            fi, xs = np.histogramdd(x[spikes[:, i] >= 1], bins=bins, range=lim, density=False)
            if sigma > 0: # PSTH
                fi = gaussian_filter(fi, sigma=sigma, mode='constant')
            f.append(fi)
    return np.stack(f, axis=-1), np.stack(xs, axis=-1)

def ms_cluster(samples, bandwidth=0.2, min_bin_freq=20, min_cluster_size=30,
               ignore_range=0.95, plot=True):
    """Create clusters of firing field clusters with mean shift.
 
    Returns
    -------
    unique_labels : surviving cluster labels
    labels        : per-sample labels (rejected clusters set to -1)
    centers       : (m, d) array of surviving cluster centres
                    (empty (0, d) array if none survive)
 
    """
    clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True,
                          cluster_all=False, min_bin_freq=min_bin_freq)
    labels = clusterer.fit_predict(samples)
    d = samples.shape[1]
    unique_labels = np.unique(labels)
    centers, to_del = [], []
    for l in unique_labels:
        if l == -1:
            continue
        members = labels == l
        if members.sum() < min_cluster_size:        # noise cluster
            to_del.append(l)
            continue
        m = samples[members].mean(axis=0)
        if np.any(np.abs(m) > ignore_range):
            labels[members] = -1
            to_del.append(l)
            continue
        centers.append(m)
    unique_labels = np.array([l for l in unique_labels
                              if l != -1 and l not in to_del])
    centers = np.stack(centers, axis=0) if centers else np.empty((0, d))
    return unique_labels, labels, centers

def best_plane(ac, az_precision=100, al_precision=50, al_max=np.pi,
               radial_fraction=RADIAL_FRACTION, radial_method="mean"):
    """Find the orientation of the autocorrelogram's best hexagonal plane.
 
    Parameters
    ----------
    ac : (d, d, d) or (d, d, d, n) non-negative autocorrelation cube(s)
 
    Returns
    -------
    az_best : (n,) azimuth (radians) of the best plane per cell
    al_best : (n,) altitude (radians) of the best plane per cell
    hgs_max : (n,) the hexagonal grid score at that best plane
    """
    hgs_map, _ = gridness_map(ac, az_precision=az_precision, al_precision=al_precision,
                              al_max=al_max, radial_fraction=radial_fraction,
                              radial_method=radial_method)   # (az, al, n)
    azs = np.linspace(0, np.pi * 2, num=az_precision, endpoint=False)
    als = np.linspace(0, al_max, num=al_precision, endpoint=False)
    n = hgs_map.shape[-1]
    az_best = np.zeros(n)
    al_best = np.zeros(n)
    hgs_max = np.zeros(n)
    for k in range(n):
        flat = hgs_map[..., k]
        i, j = np.unravel_index(np.argmax(flat), flat.shape)
        az_best[k] = azs[i]
        al_best[k] = als[j]
        hgs_max[k] = flat[i, j]
    return az_best, al_best, hgs_max


def spatial_info(p, f):
    p = p[..., None]
    f_ = f / (f * p).sum(axis=(0, 1, 2))
    f_[f_ == 0] = 1 
    return (np.log2(f_) * p * f_).sum(axis=(0, 1, 2))


def sparsity_idx(p, f):
    p = p[..., None]
    e_f = (f * p).sum(axis=(0, 1, 2))
    e_f2 = (f**2 * p).sum(axis=(0, 1, 2))
    return e_f**2 / e_f2


def si_shuffle(p, x, spikes, bins=30, sigma=1.5, N=50):
    sinfo_sf = np.zeros((N, spikes.shape[-1]))
    sidx_sf = np.zeros((N, spikes.shape[-1]))
    for i in range(N):
        spikes_sf = [np.random.permutation(spikes[:, j]) for j in range(spikes.shape[-1])]
        spikes_sf = np.stack(spikes_sf, axis=-1)
        f_sf, _ = hist3d(x, spikes_sf, bins=bins, sigma=sigma)
        sinfo_sf[i] = spatial_info(p, f_sf)
        sidx_sf[i] = sparsity_idx(p, f_sf)
    return sinfo_sf, sidx_sf


def autocorr_radial(ac, rmax, method='mean'):
    ''' Planar autocorrelation as a function of distance to the center
    '''
    # Creating coordinates
    radius = int(np.floor(len(ac)/2))
    x = np.arange(-radius, radius+1)
    x = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1)

    corr = np.zeros(rmax)
    corr[0] = ac[radius, radius] # Center

    for r in range(1, rmax):
        idxi = x[..., 0]**2 + x[..., 1]**2 > (r-1)**2
        idxo = x[..., 0]**2 + x[..., 1]**2 <= r**2
        idx = idxi & idxo

        if method == 'median':
            corr[r] = np.median(ac[idx])
        elif method == 'mean':
            corr[r] = ac[idx].mean()
        elif method == 'max':
            corr[r] = ac[idx].max()
        else: # quantile
            corr[r] = np.quantile(ac[idx], method)

    return corr


def autocorr_radial3d(ac, rmax, method='mean'):
    radius = int(np.floor(len(ac)/2))
    x = np.arange(-radius, radius+1)
    x = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1)

    corr = np.zeros(rmax)
    corr[0] = ac[radius, radius, radius] # Center

    for r in range(1, rmax):
        idxi = x[..., 0]**2 + x[..., 1]**2 + x[..., 2]**2 > (r-1)**2
        idxo = x[..., 0]**2 + x[..., 1]**2 + x[..., 2]**2 <= r**2
        idx = idxi & idxo

        if method == 'median':
            corr[r] = np.median(ac[idx])
        elif method == 'mean':
            corr[r] = ac[idx].mean()
        elif method == 'mean_comp':
            corr[r] = ac[idx].sum() / np.sqrt(len(ac[idx]))
        elif method == 'max':
            corr[r] = ac[idx].max()
        else: # quantile
            corr[r] = np.quantile(ac[idx], method)
    return corr


def peak(corr, plane, az, al, width=3, rel_height=0.75):
    ''' Find the first and second peaks
    Returns
    -------
    peaks : sequence
        peaks[0] is the radius of the center peak
        peaks[1] consists of the left and right ends of the second peak
    '''
    corr = np.hstack((corr[-1:0:-1], corr)) # Aux
    res = find_peaks(corr, width=width, rel_height=rel_height)
    peaks = res[0]
    assert len(peaks) % 2 == 1

    c = len(peaks) // 2
    p0r = res[1]['right_bases'][c] - peaks[c]

    if len(peaks) == 1:
        return [p0r, []]

    p1l = res[1]['left_bases'][c+1] - peaks[c]
    p1r = res[1]['right_bases'][c+1] - peaks[c]

    return [p0r, [p1l, p1r]]


def gridness(ac, lb, ub, hex_only=False):
    '''
    parameters
    ----------
    ac : np.ndarray
        Planar autocorrelation
    lb : int
        Lower bound
    ub : int
        Inclusive upper bound

    Returns
    -------
    hgs : float
        Hexagonal gridness score
    sgs : float
        Square gridness score
    '''
    assert ac.min() >= -1e-10
    radius = int(np.floor(len(ac)/2))
    x = np.arange(-radius, radius+1)
    x = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1)

    idxi = x[..., 0]**2 + x[..., 1]**2 > lb**2
    idxo = x[..., 0]**2 + x[..., 1]**2 <= ub**2
    idx = idxi & idxo

    im = ac.copy() # Process as an image
    im[~idx] = 0
    im = im[radius-ub:radius+ub+1, radius-ub:radius+ub+1] # Clip
    im_flat = im.flatten()

    gs = [0, 0]
    for i, a in enumerate((30, 45)):
        gsmin = min(pearsonr(im_flat, rotate(im, a*2, reshape=False).flatten())[0],
                    pearsonr(im_flat, rotate(im, a*4, reshape=False).flatten())[0])

        gsmax = max(pearsonr(im_flat, rotate(im, a*1, reshape=False).flatten())[0],
                    pearsonr(im_flat, rotate(im, a*3, reshape=False).flatten())[0],
                    pearsonr(im_flat, rotate(im, a*5, reshape=False).flatten())[0])

        gs[i] = gsmin - gsmax
        if hex_only: #possibly only need to caluclate 1 not for i
            break                          
    return gs[0], gs[1]


def gridness_map(ac, az_precision=100, al_precision=50, al_max=np.pi,
                 radial_fraction=RADIAL_FRACTION, radial_method="mean", hex_only=False):
    """Hexagonal/square gridness over a grid of (azimuth, altitude) planes.
    """
    assert ac.min() >= 0
    assert (ac.shape[0] == ac.shape[1]) and (ac.shape[2] == ac.shape[1]) \
            and (ac.shape[0] == ac.shape[2])
    if len(ac.shape) == 3:
        ac = ac[..., None]

    d, n = len(ac), ac.shape[-1]

    hgs_map = np.zeros((az_precision, al_precision, n))
    sgs_map = np.zeros((az_precision, al_precision, n))
    azs = np.linspace(0, np.pi * 2, num=az_precision, endpoint=False)
    als = np.linspace(0, al_max, num=al_precision, endpoint=False)

    for i, az in enumerate(azs):
        for j, al in enumerate(als):
            plane = oblique_slice(ac, az, al)
            plane[plane < 0] = 0
            for k in range(n):
                corr_radial = autocorr_radial(plane[..., k], int(d * radial_fraction),
                                              method=radial_method)
                res = peak(corr_radial, plane, az, al)[1]
                if len(res) == 0:
                    continue
                hgs, sgs = gridness(plane, *res, hex_only=hex_only)

                hgs_map[i, j, k] = hgs
                sgs_map[i, j, k] = sgs
    return hgs_map, sgs_map


def chi_score(ac, azimuths_deg=CHI_AZIMUTHS_DEG, altitudes_deg=CHI_ALTITUDES_DEG,
              template=CHI_TEMPLATE, radial_fraction=RADIAL_FRACTION,
              radial_method="mean"):
    """Structure scores (chi_fcc, chi_hcp, chi_col) for an autocorrelation cube.
    """
    assert ac.min() >= 0
    assert (ac.shape[0] == ac.shape[1]) and (ac.shape[2] == ac.shape[1]) \
            and (ac.shape[0] == ac.shape[2])
    if len(ac.shape) == 3:
        ac = ac[..., None]

    d, n = len(ac), ac.shape[-1]

    azs = np.asarray(azimuths_deg) * np.pi / 180
    als = tuple(a * np.pi / 180 for a in altitudes_deg)
    hgs_map = np.zeros((len(azs), len(als), n))
    sgs_map = np.zeros((len(azs), len(als), n))

    for i, az in enumerate(azs):
        for j, al in enumerate(als):
            plane = oblique_slice(ac, az, al)
            plane[plane < 0] = 0
            for k in range(n):
                corr_radial = autocorr_radial(plane[..., k], int(d * radial_fraction),
                                              method=radial_method)
                res = peak(corr_radial, plane, az, al)[1]
                if len(res) == 0:
                    continue
                hgs, sgs = gridness(plane, *res)

                hgs_map[i, j, k] = hgs
                sgs_map[i, j, k] = sgs
    f_hgs, f_sgs = template["fcc"]
    h_hgs, h_sgs = template["hcp"]
    c_hgs, c_sgs = template["col"]
    chi_fcc = np.median(hgs_map[f_hgs, 0], axis=0) + np.median(sgs_map[f_sgs, 1], axis=0)
    chi_hcp = np.median(hgs_map[h_hgs, 0], axis=0) + np.median(sgs_map[h_sgs, 1], axis=0)
    chi_col = np.median(hgs_map[c_hgs, 0], axis=0) + np.median(sgs_map[c_sgs, 1], axis=0)
    return chi_fcc, chi_hcp, chi_col


def gen_hex_layer(center, r, rotz, bbox=(-1, 1)):
    points = [center]
    lb, ub = 0, 1
    for k in range(round(np.sqrt(8) / r)):
        for center in points[lb:ub]:
            for i in range(6):
                d = i * np.pi / 3 + np.pi / 6 + rotz
                pt = center + r * np.array([np.cos(d), np.sin(d)])
                if (pt < bbox[0]).any() or (pt > bbox[1]).any():
                    continue
                if (lb > 0) and (np.linalg.norm(pt[None, :] - points, axis=1) < 1e-3).any():
                    continue
                points.append(pt)
        lb = ub
        ub = len(points)
    return np.array(points)


def hexagonal_structure(struct_type, r, rotz):
    layera = gen_hex_layer(np.zeros(2), r, rotz, bbox=(-1.5, 1.5))
    d = np.array([r / np.sqrt(3) * np.cos(rotz), r / np.sqrt(3) * np.sin(rotz)])
    layerb = gen_hex_layer(d, r, rotz, bbox=(-1.5, 1.5))
    if struct_type == 'fcc':
        layers, h = (layera, -layerb, layerb), np.sqrt(6) / 3 * r
    elif struct_type == 'hcp':
        layers, h = (layera, -layerb), np.sqrt(6) / 3 * r
    elif struct_type == 'col':
        layers, h = (layera,), r/10
    else:
        raise ValueError

    centers = []
    zs = np.arange(-1.2, 1.2, h)
    for i, z in enumerate(zs):
        l = layers[i % len(layers)]
        centers.append(np.hstack((l, np.ones((l.shape[0], 1)) * z)))

    centers = rot_x(np.vstack(centers), np.pi / 180 * rotz, 0)
    return centers