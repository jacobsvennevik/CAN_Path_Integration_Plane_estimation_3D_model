"""
Extending the visualization module of the MADE package for manifolds, CANs, and QANs.
Because it is hard to visualize a 4D object (3D volume plotted in one more dimension).
We break the 3D space into 2D volumes. Look in notebooks for simulations

This module provides functions to visualize:
1. Manifold geometries and distances
2. CAN connectivity and states
3. QAN trajectories and states
"""
from made.can import CAN
from made.qan import QAN
from made.visuals import clean_axes

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

def _plot_slice(ax, X, Y, sl, ref_xy, xlabel, ylabel, title,
                cmap, vmin, vmax, ref_color="black", ref_label="Selected neuron"):
    """
    Plot a single 2D slice as a filled contour on ax.

    Args:
        ax:        Matplotlib Axes to draw on
        X, Y:      2D coordinate grids for the two varying dimensions
        Z:         2D values to contour (connectivity weight or distance)
        ref_xy:    (x, y) position of the reference point in this slice
        xlabel:    Label for x axis
        ylabel:    Label for y axis
        title:     Axes title
        cmap:      Colormap
        vmin/vmax: Colormap limits (pass None for auto-scaling)
        ref_color: Marker color for reference point
        ref_label: Legend label for reference point
    """
    # Create a contour plot
    contour = ax.contourf(
        X,
        Y,
        sl,
        levels=50,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax)

    plt.colorbar(contour, ax=ax)
    #plot the neurons in a scatter plot
    ax.scatter(ref_xy[0], ref_xy[1], color=ref_color, s=100, marker="*", label=ref_label)
    ax.legend()
    clean_axes(ax, title=title, ylabel=ylabel)
    ax.set_xlabel(xlabel)

def _scatter_plot(ax, X, Y, Z, ref_xy, xlabel, ylabel, title, cmap, vmin, vmax):
    """
    Plot a single 2D slice as a scatterplot
    """
    scatter = ax.scatter(
        X.ravel(), Y.ravel(), c=Z.ravel(),
        cmap=cmap, vmin=vmin, vmax=vmax, s=15,
        )
    plt.colorbar(scatter, ax=ax, label="Neuron state")
    clean_axes(ax, title=title, ylabel=ylabel)
    ax.set_xlabel(xlabel)
    
def _neuron_counts(can):
    """ Calculate grid dimensions based on spacing (number of nerons along theta_i) """
    return can.nx(0), can.nx(1), can.nx(2) #neurons along theta_1


def _visualize_3d_data_slices(can, data_3d, ref_idx, title_prefix, cmap,
                               vmin=None, vmax=None, plot_fn=None):
    """
    Helper function render three axis-aligned 2D slices of a (n1×n2×n3) volume.

    Args:
        can:          CAN instance (provides grid dimensions and neuron coordinates)
        data_3d:      Shape (n1, n2, n3) array to visualize
        ref_idx:      Flat neuron index that determines the slice positions
        title_prefix: String prepended to each subplot title
        cmap:         Colormap
        vmin/vmax:    Colormap limits (pass None for auto-scaling per slice)

    Returns:
        fig, axes: figure and length-3 axes array
    """
    #how man neurons pr dimensions
    n1,n2,n3 = _neuron_counts(can)
    
    #change the flat neuron coordinate array into a 3D grid
    coords_3d  = can.neurons_coordinates.reshape(n1, n2, n3, 3)
    #converts the ref_idx (for the flat arrray), into index for coords_3d
    i1, i2, i3 = np.unravel_index(ref_idx, (n1, n2, n3))
    #retrives the coordinates in radiens   
    ref_coords  = can.neurons_coordinates[ref_idx]

    #Slice the 3D torus into 3 planes, keeping the last one fixed
    slices = [
        #Horizontal Plane THETA_1 vs THETA_2
        (coords_3d[:, :, i3, 0], #take THETA_1 indices (0), fix theta_3 (i3), becomes X for the plot
         coords_3d[:, :, i3, 1], #take THETA_2 indices (1), fix theta_3 (i3), becomes Y for the plot
         data_3d[:, :, i3], #This is the connecitivty between neuruons above, becomes Z the colours.
         (ref_coords[0], ref_coords[1]), #the position of reference neuron (star)
         "theta_1", "theta_2", f"{title_prefix}, fix θ₃={ref_coords[2]:.2f}"), #titels 
        #THETA_2 vs THETA_3
        (coords_3d[i1, :, :, 1], coords_3d[i1, :, :, 2], data_3d[i1, :, :],
         (ref_coords[1], ref_coords[2]), "theta_2", "theta_3",
         f"{title_prefix}, fix θ₁={ref_coords[0]:.2f}"),
        #THETA_1 vs THETA_3
        (coords_3d[:, i2, :, 0], coords_3d[:, i2, :, 2], data_3d[:, i2, :],
         (ref_coords[0], ref_coords[2]), "theta_1", "theta_3",
         f"{title_prefix}, fix θ₂={ref_coords[1]:.2f}"),
    ]

    if plot_fn is None:
        plot_fn = _plot_slice

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (X, Y, Z, ref_xy, x_label, y_label, title) in zip(axes, slices):
        plot_fn(ax, X, Y, Z, ref_xy, x_label, y_label, title,
                cmap=cmap, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    return fig, axes


def _visualize_conn_3d(can, neuron_idx, cmap="bwr", vmin=-1, vmax=0):
    """
    Visualise Can connecitivty for a single neuron.
    This is done by slicing the 3D manifold into 2D manifolds and plotting each of them.

    Args:
        ax: The matplotlib axes to plot on
        can: The CAN instance
        neuron_idx: Index of the neuron whose connectivity to visuali3e
        cmap: Colormap for connectivity values
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling

    Returns:
        The matplotlib axes with the plot
    """
    #how man neurons pr dimensions
    n1,n2,n3 = _neuron_counts(can)
    
    # Reshape flat connecitivty matrix and neuron coordinates back into 3D grid
    conn_3d = can.connectivity_matrix[neuron_idx].reshape(n1, n2, n3)
    # Delegate to the generic slicer and anchors all three slices at neuron_idx
    return _visualize_3d_data_slices(
        can, conn_3d, neuron_idx, "Connectivity", cmap=cmap, vmin=vmin, vmax=vmax
    )


def _visualize_manifold_3d(mfld, show_distances=False, distance_point=None, cmap="Greens"):
    """
    Visualize a 3D manifold (T^3) as three 2D slices.
    Distance is computed using the full 3D metric.
    Shadowing the same function in the MADE package.

    Args:
        mfld:           3D manifold with .metric and .parameter_space
        show_distances: Whether to overlay distance contours
        distance_point: Shape (3,) reference point
        cmap:           Colormap for distance values

    Returns:
        fig, axes: figure and length-3 axes array
    """
    if show_distances and distance_point is None:
        raise ValueError("distance_point must be provided when show_distances=True")

    #sample points per dimension
    n = 50
    param_space = mfld.parameter_space

    #lookup table for what is varying in each slice.
    slice_specs = [
        ((0, 1), 2, "theta_1", "theta_2"),
        ((0, 2), 1, "theta_1", "theta_3"),
        ((1, 2), 0, "theta_2", "theta_3"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax_i, ((d0, d1), d_fixed, x_label, y_label) in zip(axes, slice_specs):

        if show_distances:
            # picks points from the full 3D parameter space. Giving a complete grid (theta_1, theta_2, theta_3,)
            points = param_space.sample(n)
            #Change it to 2D
            # and copy so that we dont change the orginal 3D grid
            query  = points.copy()
            query[:, d_fixed] = distance_point[d_fixed]

            #compute the distance from every point in the slice to the reference point.
            distances = mfld.metric(query, distance_point.reshape(1, -1))

            # Reshape the long list of n^3 points into an (n×n×n) cube
            # so we can get the 2D slices by fixing one axis
            coords_3d = query.reshape(n, n, n, 3)
            dist_3d   = distances.reshape(n, n, n)
            mid = n // 2

            if d_fixed == 0:
                h_coords = coords_3d[mid, :, :, d0]   # horizontal axis coordinates
                v_coords = coords_3d[mid, :, :, d1]   # vertical axis coordinates
                distances_2d = dist_3d[mid, :, :]     # distances between every point to the reference point

            elif d_fixed == 1:
                h_coords = coords_3d[:, mid, :, d0]
                v_coords = coords_3d[:, mid, :, d1]
                distances_2d = dist_3d[:, mid, :]

            elif d_fixed == 2:
                h_coords = coords_3d[:, :, mid, d0]
                v_coords = coords_3d[:, :, mid, d1]
                distances_2d = dist_3d[:, :, mid]
            else:
                raise ValueError(f"d_fixed must be 0, 1, or 2, got {d_fixed}.")

            #reference point for title
            fixed_val = distance_point[d_fixed]
            title = f"Fix dim{d_fixed}={fixed_val:.2f}"
            ## position of the reference point projected onto this 2D panel
            ref_xy = (distance_point[d0], distance_point[d1])

            _plot_slice(ax_i, h_coords, v_coords, distances_2d, ref_xy, x_label, y_label, title,
                        cmap=cmap, vmin=None, vmax=None,
                        ref_color="red", ref_label="Reference point")

        else:
            # No distances: render labeled empty axes as a placeholder
            clean_axes(ax_i, title=f"dim{d_fixed} fixed", ylabel=y_label)
            ax_i.set_xlabel(x_label)

    fig.suptitle(f"{mfld.__class__.__name__} — axis-aligned slices", y=1.02)
    plt.tight_layout()
    return fig, axes


# ------------------------------------------------------------------ #
#                VISUALIZE CAN CONNECTIVITY (3D)                     #
# ------------------------------------------------------------------ #

def visualize_can_connectivity_3d(can, cmap="bwr", vmin=-1, vmax=0):
    """Visualize how a neuron is connected to it´s neighbours

    Done by mirroring the pattern of visualize_can_connectivity from made.visuals.py
    Selects 4 random neurons and shows
    three connectivity slices per neuron.

    Args:
        can: The CAN instance whose manifold has dim == 3
        cmap: Colormap for connectivity values
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling

    Returns:
        list of (fig, axes) tuples, one per selected neuron
    """
    assert can.manifold.dim == 3, (
        f"visualize_can_connectivity_3d requires a 3D CAN, got dim={can.manifold.dim}."
    )
    #Picking the random neurons from the total population
    total_neurons = can.neurons_coordinates.shape[0]
    neuron_idxs = np.random.choice(total_neurons, 4, replace=False)
    results = []
    #creating a plot for each neuron with each neuron plotted 3 times (2D slices)
    for neuron_idx in neuron_idxs:
        fig, axes = _visualize_conn_3d(can, neuron_idx, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.suptitle(f"Neuron {neuron_idx}", y=1.02)
        results.append((fig, axes))
    plt.tight_layout()
    return results


def visualize_can_state_3d(can, cmap="inferno"):
    """Visualize the current activity state of a 3D CAN as three 2D slices.

    Slices through the neuron with peak activation, making the bump clearly
    visible in all three axis-aligned panels. Each neuron is rendered as an
    individual dot (scatter) so the discrete neuron structure is visible.

    Args:
        can:  CAN instance whose manifold has dim == 3
        cmap: Colormap for neuron activation values

    Returns:
        fig, axes: figure and length-3 axes array
    """
    #how man neurons pr dimensions
    n1,n2,n3 = _neuron_counts(can)
    
    state_3d = can.S.reshape(n1, n2, n3)
    #returns the peak activation
    max_idx = int(can.S.argmax())

    return _visualize_3d_data_slices(can, state_3d, max_idx, "State", cmap=cmap,
                                     plot_fn=_scatter_plot)

#Todo: Not sure if this is in use anymore
def _plot_marginals(data, title, color, alpha, s):
    """Plot 2D marginal projections of 3D data."""
    
    pairs = [
    (0, 1, 'θ₁', 'θ₂', 'Marginal: all θ₃'),
    (0, 2, 'θ₁', 'θ₃', 'Marginal: all θ₂'),
    (1, 2, 'θ₂', 'θ₃', 'Marginal: all θ₁'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (i, j, li, lj, panel_title) in zip(axes, pairs):
        ax.scatter(data[:, i], data[:, j], color=color, alpha=alpha, s=s)
        ax.set_xlabel(li)
        ax.set_ylabel(lj)
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, 2 * np.pi)
        ax.set_title(panel_title)
        ax.set_aspect('equal')
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()




def wrapped_angle_diff(a, b, period=2 * np.pi):
    """
    Since theta_i lives on the circle (is periodic) we need to calculate the angular distance.  
    """
    return (a - b + period / 2) % period - period / 2


def isomap_slice(
    final_states,
    bump_coords,
    target_theta3,
    slice_width,
    n_pca=50,
    n_neighbors=8,
):
    """
    Takes a slice of the T^3 manifold at a fixed θ₃ value and
    visualizes the T² slice using dimensionality reduction with PCA and Isomap.
    """
    #Slicing logic
    #take the final simulations states, filter so that only the states inside the slice_width is added 
    dtheta3 = wrapped_angle_diff(bump_coords[:, 2], target_theta3)
    mask = np.abs(dtheta3) < slice_width
    #Apply the mask
    states_slice = final_states[mask] #the full population vectors
    coords_slice = bump_coords[mask] #decoded positions

    # Remove duplicated settled states so that identical bumps don't dominate later analysis
    # Widt larger amount of neurons might not be needed
    _, first_idx = np.unique(
        np.round(coords_slice, 6), axis=0, return_index=True,
    )
    states_slice = states_slice[first_idx]
    coords_slice = coords_slice[first_idx]
    #Run the PCA reduction and ISO embedding
    states_pca = PCA(n_components=n_pca).fit_transform(states_slice)
    iso = Isomap(
        n_components=3, n_neighbors=n_neighbors,
    )
    embedding = iso.fit_transform(states_pca)
    #How much information is lost
    print(f"reconstruction error {iso.reconstruction_error():.4f}")
    return embedding, coords_slice


def plot_isomap_slices(
    final_states,
    bump_coords,
    can,
    n_slices=4,
    slice_width=None,
    elev=30,
    azim=45,
    point_color="black",
    point_alpha=0.75,
    point_size=18,
):
    """
    Plot PCA to Isomap embeddings of T^2 slices.
    """
    if slice_width is None:
        slice_width = can.spacing

    n3 = can.nx(2)
    indices = sorted({
        max(1, int(round((i + 1) * n3 / (n_slices + 1))))
        for i in range(n_slices)
    })
    theta3s = [can.idx2coord(i, 2) for i in indices]

    fig = plt.figure(figsize=(5 * len(theta3s), 5))
    axes = []
    for col, theta3 in enumerate(theta3s):
        embedding, _ = isomap_slice(
            final_states=final_states,
            bump_coords=bump_coords,
            target_theta3=theta3,
            slice_width=slice_width,
        )
        ax = fig.add_subplot(1, len(theta3s), col + 1, projection="3d")
        axes.append(ax)

        if embedding is None:
            ax.set_title(f"\u03b8\u2083 = {theta3:.2f}\nnot enough data")
            continue

        ax.scatter(
            embedding[:, 0], embedding[:, 1], embedding[:, 2],
            color=point_color, alpha=point_alpha, s=point_size,
        )
        ax.set_title(f"\u03b8\u2083 = {theta3:.2f}")
        ax.set_aspect("equal")
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(
        "T\u00b3 CAN: PCA \u2192 Isomap visualizations of T\u00b2 slices",
        fontsize=12,
    )
    plt.tight_layout()
    return fig, axes

