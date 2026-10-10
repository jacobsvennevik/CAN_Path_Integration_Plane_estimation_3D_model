"""
Gong & Yu's scoring logic, pasted almost verbatum to keep consistency.
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import rotate
from scipy.signal import find_peaks
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import MeanShift

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
                 radial_fraction=RADIAL_FRACTION, radial_method="mean", hex_only=False,
                 azimuths_deg=None, altitudes_deg=None, return_ring=False):
    """Turns one 3D autocorrelogram into two score maps, one hexagonal and one square. 
    This is done by cutting a plane through the center at every sampled orientation and 
    scoring that plane as a flat grid.
    """
    assert (ac.shape[0] == ac.shape[1]) and (ac.shape[2] == ac.shape[1]) \
            and (ac.shape[0] == ac.shape[2])
    if len(ac.shape) == 3:
        ac = ac[..., None]

    d, n = len(ac), ac.shape[-1]
    if azimuths_deg is None:
        azs = np.linspace(0, np.pi * 2, num=az_precision, endpoint=False)
    else:
        azs = np.asarray(azimuths_deg, dtype=float) * np.pi / 180.0
    if altitudes_deg is None:
        als = np.linspace(0, al_max, num=al_precision, endpoint=False)
    else:
        als = np.asarray(altitudes_deg, dtype=float) * np.pi / 180.0

    hgs_map = np.zeros((len(azs), len(als), n))
    sgs_map = np.zeros((len(azs), len(als), n))
    ring = np.zeros((len(azs), len(als), n), dtype=bool)

    for i, az in enumerate(azs):
        for j, al in enumerate(als):
            plane = oblique_slice(ac, az, al)
            for k in range(n):
                corr_radial = autocorr_radial(plane[..., k], int(d * radial_fraction),
                                              method=radial_method)
                try:
                    res = peak(corr_radial, plane, az, al)[1]
                except (AssertionError, ValueError):
                    hgs_map[i, j, k] = np.nan
                    sgs_map[i, j, k] = np.nan
                    continue
                if len(res) == 0:
                    continue
                hgs, sgs = gridness(plane, *res, hex_only=hex_only)
                hgs_map[i, j, k] = hgs
                sgs_map[i, j, k] = sgs
                ring[i, j, k] = True
    if return_ring:
        return hgs_map, sgs_map, ring
    return hgs_map, sgs_map

