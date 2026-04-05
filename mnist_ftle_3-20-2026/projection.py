from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def project_to_2d(x: np.ndarray, method: str = "pca") -> np.ndarray:
    if method != "pca":
        raise ValueError(f"Unsupported projection method: {method}")
    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(x)
