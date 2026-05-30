import numpy as np
from made.metrics import Metric

class HexagonalPeriodicMetric(Metric):
    """
    Make a periodic twisted metric, to get the hexagonal lattice in 2D.
    Uses the twisted torus logic from Burak and Fiete (2009) and Guanella (2007).
    Intuitivly, this is what glues the edges of the nerual sheets together so that the torus is periodic.  
    """

    def __init__(self, dim: int = 3, period: float = 2 * np.pi):
        if dim < 2:
            raise ValueError("TwistedTorus needs at least 2 dimensions")
        self.dim = dim #dimensions
        self.period = period
        self.B_unit = np.array([[1.0, 0.5],
                                [0.0, np.sqrt(3) / 2.0]])  # the actual geometry, how we get twisted connection
        self.B_inv  = np.linalg.inv(self.B_unit) #inverse basis matrix
        self.B      = period * self.B_unit # scaled basis matrix 
        self._nb    = np.array([[i, j]               # Declreaes the topology. TODO: Check if we should make z also included here now X and Y is but not Z
                            for i in (-1, 0, 1)
                            for j in (-1, 0, 1)])  

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        diff = x - y

        # Finds the minimum hexagonal distance on the manifold.
        f = diff[:, :2] / self.period
        f = f - np.round(f)                                  # nearest cell
        cand = f[:, None, :] + self._nb[None, :, :]          # (n, 9, 2) candidate images
        cart = cand @ self.B.T                               # (n, 9, 2) Cartesian vectors
        d2_hex = np.min(np.sum(cart ** 2, axis=2), axis=1)   # (n,) squared shortest dist

        # remaining axes (Z) is just periodic
        if self.dim > 2:
            rest = diff[:, 2:]
            rest = np.mod(rest + self.period / 2, self.period) - self.period / 2
            d2_rest = np.sum(rest ** 2, axis=1)
        else:
            d2_rest = 0.0

        return np.sqrt(d2_hex + d2_rest)

    def pairwise_distances(self, X, weights_offset=lambda x: x):
        """ Build the fill pairwise distance matrix between all neurons. 
        
        """
        X_transformed = weights_offset(X.copy())
        n = X.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            D[i, :] = self(X[i:i + 1], X_transformed)
        return D
    
    def to_phase(self, v: np.ndarray) -> np.ndarray:
        """Physical (Cartesian) to lattice phase (twisted).
        Applies B_inv to the first two axes; leaves any further axes unchanged."""
        v = np.asarray(v, dtype=float).copy()
        v[..., :2] = v[..., :2] @ self.B_inv.T
        return v

    def to_physical(self, theta: np.ndarray) -> np.ndarray:
        """Lattice phase (twisted) to physical (Cartesian).
        Inverse of to_phase. Used when you need decoded position in metres."""
        t = np.asarray(theta, dtype=float).copy()
        t[..., :2] = t[..., :2] @ self.B_unit.T
        return t