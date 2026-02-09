import numpy as np
import deepchem as dc
from deepchem.models.symbolic_regression import SymbolicRegressionModel

# Load a real dataset
tasks, (train, valid, test), transformers = dc.molnet.load_delaney(
    featurizer="rdkit",
    splitter="random",
    transformers=[]
)

def shrink(ds, n_rows=200, n_feats=6, seed=123):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=min(n_rows, len(ds)), replace=False)
    X = ds.X[idx, :n_feats].astype(np.float32)
    y = ds.y[idx].astype(np.float32).reshape(-1)
    return dc.data.NumpyDataset(X, y)


train_small = shrink(train)
test_small = shrink(test, n_rows=200, n_feats=6, seed=321)




model = SymbolicRegressionModel(
    ops=["add", "mul"],
    unary_ops=[],
    binary_ops=["add", "mul"],
    niterations=6,
    population_size=35,
    populations=1,
    ncycles_per_iteration=30,
    maxsize=8,
    maxdepth=6,
    tree_depth=3,
    optimizer_probability=1.0,
    should_optimize_constants=True,
    opt_steps=80,
    opt_lr=0.1,
    seed=123,
    use_multiprocessing=False,
)

model.fit(train_small)
print("Best equation:", model.get_best_expression())


preds = model.predict(test_small)[:, 0]
mse = np.mean((preds - test_small.y.reshape(-1)) ** 2)
print("Test MSE:", mse)
