"""Instance primitives: coordinate-uniqueness, binary I/O, distribution registry."""
import json
import struct
from typing import Any, Dict

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def make_unique_numba(coords: np.ndarray, grid_size: float, seed: int) -> np.ndarray:
    """Dedup duplicate points by perturbing collisions within a small tolerance."""
    np.random.seed(seed)
    n, d = coords.shape
    max_retries = 10
    tolerance = 1e-6

    for attempt in range(max_retries):
        unique = True
        for i in range(n - 1):
            for j in range(i + 1, n):
                same = True
                for k in range(d):
                    if abs(coords[i, k] - coords[j, k]) > tolerance:
                        same = False
                        break
                if same:
                    unique = False
                    for k in range(d):
                        coords[j, k] += np.random.uniform(-grid_size * 1e-5, grid_size * 1e-5)
                        coords[j, k] = max(0.0, min(grid_size, coords[j, k]))
                    break
            if not unique:
                break

        if unique:
            return coords

    return coords


def save_instance_binary(instance_path: str, data: Dict[str, Any]) -> None:
    binary_path = instance_path.replace(".json", ".bin")
    n = data["n_customers"]
    d = data["dimension"]
    coords = np.array(data["coordinates"], dtype=np.float32)

    with open(binary_path, "wb") as f:
        f.write(struct.pack("III", n, d, data["grid_size"]))
        dist_str = "".join(data["distribution_types"])
        f.write(struct.pack("I", len(dist_str)))
        f.write(dist_str.encode("ascii"))
        f.write(coords.tobytes())

    with open(instance_path, "w") as f:
        json.dump(data, f)


def load_instance_binary(binary_path: str) -> Dict[str, Any]:
    with open(binary_path, "rb") as f:
        n, d, grid_size = struct.unpack("III", f.read(12))
        dist_len = struct.unpack("I", f.read(4))[0]
        dist_str = f.read(dist_len).decode("ascii")
        coords = np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d)

    return {
        "n_customers": n,
        "dimension": d,
        "grid_size": grid_size,
        "distribution_types": list(dist_str),
        "coordinates": coords.tolist(),
    }


class DistributionGenerator:
    def __init__(self):
        self.cache = {}

    def get_rng(self, seed: int) -> np.random.Generator:
        if seed not in self.cache:
            self.cache[seed] = np.random.default_rng(seed)
        return self.cache[seed]

    def generate_1d_random(self, n: int, seed: int, grid_size: float, **kwargs) -> np.ndarray:
        return self.get_rng(seed).uniform(0, grid_size, n)

    def generate_1d_normal(self, n: int, seed: int, grid_size: float, **kwargs) -> np.ndarray:
        rng = self.get_rng(seed)
        result = rng.normal(loc=grid_size / 2, scale=grid_size / 6, size=n)
        np.clip(result, 0, grid_size, out=result)
        return result

    def generate_1d_clustered(self, n: int, seed: int, grid_size: float, **kwargs) -> np.ndarray:
        rng = self.get_rng(seed)
        clust_n = max(2, int(np.sqrt(n) / 4))
        centers = rng.uniform(0, grid_size, clust_n)
        assignments = rng.integers(0, clust_n, size=n)
        stdev = grid_size * 0.05
        result = rng.normal(loc=centers[assignments], scale=stdev)
        np.clip(result, 0, grid_size, out=result)
        return result

    def generate_1d_correlated(self, n: int, seed: int, grid_size: float, base=None, **kwargs) -> np.ndarray:
        rng = self.get_rng(seed)
        if base is None:
            return rng.uniform(0, grid_size, n)
        noise = rng.normal(loc=0, scale=grid_size / 10, size=n)
        result = base + noise
        np.clip(result, 0, grid_size, out=result)
        return result


dist_gen = DistributionGenerator()
DISTRIBUTION_MAP_1D = {
    "r": dist_gen.generate_1d_random, "n": dist_gen.generate_1d_normal,
    "c": dist_gen.generate_1d_clustered, "o": dist_gen.generate_1d_clustered,
    "i": dist_gen.generate_1d_normal, "p": dist_gen.generate_1d_random,
    "t": dist_gen.generate_1d_random, "g": dist_gen.generate_1d_random,
    "k": dist_gen.generate_1d_correlated, "l": dist_gen.generate_1d_random,
    "s": dist_gen.generate_1d_random, "b": dist_gen.generate_1d_random,
    "x": dist_gen.generate_1d_random, "e": dist_gen.generate_1d_random,
    "a": dist_gen.generate_1d_random, "h": dist_gen.generate_1d_random,
}
DIST_LETTERS = list(DISTRIBUTION_MAP_1D.keys())
