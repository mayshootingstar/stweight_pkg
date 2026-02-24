import numpy as np

def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance (km) using Haversine formula."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return float(R * c)

def distance_matrix_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Compute (n,n) distance matrix in km."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n = len(lon)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i, j] = 0.0
            else:
                D[i, j] = haversine_km(lon[i], lat[i], lon[j], lat[j])
    return D