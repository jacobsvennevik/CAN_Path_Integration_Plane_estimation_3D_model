import numpy as np
from made.metrics import Metric

class HexagonalPeriodicMetric(Metric):
    """
    Make a periodic twisted metric, to get the hexagonal lattice in 2D.
    Uses the twisted torus logic from Burak and Fiete (2009) and Guanella (2007).
    Intuitivly, this is what glues the edges of the nerual sheets together so that the torus is periodic.  
    """

    def __init__(self, dim=3, period=2*np.pi, lattice="fcc"):
        self.dim, self.period = dim, period
        if lattice == "hex":                              # original: hex(xy) ⊗ periodic z
            self.B_unit  = np.array([[1.0, 0.5], [0.0, np.sqrt(3)/2]])
            self._nb     = np.array([[i, j] for i in (-1,0,1) for j in (-1,0,1)])
            self._coupled = 2
        elif lattice == "fcc":
            M = np.array([[0.,1.,1.], [1.,0.,1.], [1.,1.,0.]])
            self.B_unit  = M / np.sqrt(2)                 # nn distance = 1 (matches hex)
            self._nb     = np.array([[i,j,k] for i in (-1,0,1)
                                            for j in (-1,0,1) for k in (-1,0,1)])  # 27
            self._coupled = 3
        self.B_inv = np.linalg.inv(self.B_unit)
        self.B     = period * self.B_unit

    def __call__(self, x, y):
        x, y = np.atleast_2d(x), np.atleast_2d(y)
        c = self._coupled
        f = (x - y)[:, :c] / self.period
        f = f - np.round(f)
        cart = (f[:, None, :] + self._nb[None, :, :]) @ self.B.T
        d2 = np.min(np.sum(cart**2, axis=2), axis=1)
        if self.dim > c:                                  # only fires for "hex" (c=2, dim=3)
            rest = np.mod((x-y)[:, c:] + self.period/2, self.period) - self.period/2
            d2 = d2 + np.sum(rest**2, axis=1)
        return np.sqrt(d2)

    """def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
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

        return np.sqrt(d2_hex + d2_rest)"""

    def pairwise_distances(self, X, weights_offset=lambda x: x):
        """ Build the fill pairwise distance matrix between all neurons. 
        
        """
        X_transformed = weights_offset(X.copy())
        n = X.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            D[i, :] = self(X[i:i + 1], X_transformed)
        return D
    
    def to_phase(self, v):
        v = np.asarray(v, float).copy()
        c = self._coupled
        v[..., :c] = v[..., :c] @ self.B_inv.T
        return v 

    def to_physical(self, theta: np.ndarray) -> np.ndarray:
        """Lattice phase (twisted) to physical (Cartesian).
        Inverse of to_phase. Used when you need decoded position in metres."""
        t = np.asarray(theta, dtype=float).copy()
        c = self._coupled
        t[..., :c] = t[..., :c] @ self.B_unit.T @ self.B_unit.T
        return t