from dataclasses import dataclass

import numpy as np
from cyanure.estimators import Classifier
import torch
from tqdm import tqdm

from volt import config


@dataclass
class LinearProbeConfig:
    tol: float = 1e-3
    solver: str = 'catalyst-miso'
    max_iter: int = 500
    verbose: bool = True
    random_state: int = 42
    duality_gap_interval: int = 10
    lambda_scaled: float = 0.01


config.register('linear_probe', LinearProbeConfig)


def probe(cfg, backbone, out_dim, dataloader, device, bias=True):
    backbone.eval()
    backbone = backbone.to(device)

    with torch.no_grad():
        X, y = [], []
        for x, label in tqdm(dataloader, desc='Embeddings for LP'):
            X.append(backbone(x.to(device)).cpu())
            y.append(label)
        X = torch.cat(X).numpy().astype('float64')
        y = torch.cat(y).numpy()

    classifier = Classifier(loss="multiclass-logistic", penalty="l2", fit_intercept=False,
                            tol=cfg.lp.tol,
                            solver=cfg.lp.solver,
                            lambda_1=cfg.lp.lambda_scaled / X.shape[0],
                            max_iter=cfg.lp.max_iter,
                            duality_gap_interval=cfg.lp.duality_gap_interval,
                            multi_class="ovr", verbose=cfg.lp.verbose)
    if bias:
        X = np.hstack([X, np.ones((X.shape[0], 1))])
    classifier.fit(
        X,
        y,
    )
    score = classifier.score(X, y)
    print(f"Linear probe accuracy (train): {score:.2%}")
    r = classifier.get_weights()
    if bias:
        b = r[-1]
        r = r[:-1]
        return (r, b), score
    return r, score
