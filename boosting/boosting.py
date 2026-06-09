from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor, _tree
from tqdm.auto import tqdm





class FeatureQuantizer:
    """Нужна для задания: 2.6.1, 2.6.2."""
    def __init__(self, quantization_type: str, nbins: int = 255):
        """Нужна для задания: 2.6.1, 2.6.2."""
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.edges_: list[np.ndarray | None] = []

    def _should_quantize(self, j: int, feature_types, cat_features) -> bool:
        """Нужна для задания: 2.6.1, 2.6.2."""
        if cat_features is not None and j in cat_features:
            return False
        if feature_types is not None and feature_types[j] != "real":
            return False
        return True

    @staticmethod
    def _bin_entropy(counts: np.ndarray) -> float:
        """Нужна для задания: 2.6.2."""
        total = counts.sum()
        if total <= 0:
            return 0.0
        p = counts / total
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    def _fit_column_min_entropy(self, col: np.ndarray) -> np.ndarray:
        """Нужна для задания: 2.6.2."""
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            return np.array([0.0, 1.0])
        values, counts = np.unique(finite, return_counts=True)
        if values.size <= self.nbins:
            edges = np.concatenate([[values[0] - 1e-9], values, [values[-1] + 1e-9]])
            return edges.astype(float)

        groups = [[i, i, counts[i]] for i in range(len(values))]
        while len(groups) > self.nbins:
            best_merge, best_cost = None, np.inf
            for g in range(len(groups) - 1):
                merged_counts = groups[g][2] + groups[g + 1][2]
                cost = self._bin_entropy(np.array([merged_counts]))
                if cost < best_cost:
                    best_cost = cost
                    best_merge = g
            g = best_merge
            groups[g][1] = groups[g + 1][1]
            groups[g][2] += groups[g + 1][2]
            del groups[g + 1]

        edges = [values[groups[0][0]] - 1e-9]
        for g in groups:
            left, right = g[0], g[1]
            edges.append((values[left] + values[right]) / 2)
        edges.append(values[groups[-1][1]] + 1e-9)
        return np.array(edges, dtype=float)

    def _fit_column_piecewise(self, col: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Нужна для задания: 2.6.2."""
        finite = np.isfinite(col)
        col = col[finite].astype(float)
        y = y[finite]
        if col.size == 0:
            return np.array([0.0, 1.0])
        y01 = (y == 1).astype(float) if np.unique(y).size > 1 else np.zeros_like(y)

        order = np.argsort(col)
        x = col[order]
        y_sorted = y01[order]
        mask = x[:-1] != x[1:]
        thresholds = (x[:-1][mask] + x[1:][mask]) / 2
        if thresholds.size == 0:
            lo, hi = x.min(), x.max()
            return np.array([lo, hi + 1e-9], dtype=float)

        scores = []
        ones_all = y_sorted.sum()
        n = len(y_sorted)
        ones_l_all = np.cumsum(y_sorted)[:-1][mask]
        R_l = np.arange(1, n)[mask]
        R_r = n - R_l
        ones_r_all = ones_all - ones_l_all
        zeros_l = R_l - ones_l_all
        zeros_r = R_r - ones_r_all
        for i in range(len(thresholds)):
            if R_l[i] == 0 or R_r[i] == 0:
                scores.append(-np.inf)
                continue
            gini_l = 1 - (ones_l_all[i] / R_l[i]) ** 2 - (zeros_l[i] / R_l[i]) ** 2
            gini_r = 1 - (ones_r_all[i] / R_r[i]) ** 2 - (zeros_r[i] / R_r[i]) ** 2
            scores.append(-(R_l[i] * gini_l + R_r[i] * gini_r))

        scores = np.array(scores)
        k = min(self.nbins - 1, scores.size)
        best = np.argsort(-scores)[:k]
        inner = np.sort(thresholds[best])
        edges = np.concatenate([[x.min() - 1e-9], inner, [x.max() + 1e-9]])
        return edges.astype(float)

    def _fit_column(self, col: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Нужна для задания: 2.6.1, 2.6.2."""
        col = self._column_as_float(col)
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            return np.array([0.0, 1.0])

        if self.quantization_type == "uniform":
            lo, hi = finite.min(), finite.max()
            if lo == hi:
                return np.array([lo, hi + 1e-9], dtype=float)
            return np.linspace(lo, hi, self.nbins + 1)

        if self.quantization_type == "quantile":
            qs = np.linspace(0, 1, self.nbins + 1)
            edges = np.quantile(finite, qs)
            edges = np.unique(edges)
            if edges.size < 2:
                v = edges[0]
                return np.array([v, v + 1e-9], dtype=float)
            return edges.astype(float)

        if self.quantization_type == "min_entropy":
            return self._fit_column_min_entropy(col)

        if self.quantization_type == "piecewise":
            return self._fit_column_piecewise(col, y)


    def fit(
        self,
        X: np.ndarray,
        feature_types: Iterable[str] | None = None,
        cat_features: Iterable | None = None,
        y: np.ndarray | None = None,
    ) -> FeatureQuantizer:
        """Нужна для задания: 2.6.1, 2.6.2."""
        n_features = X.shape[1]
        if feature_types is not None:
            feature_types = list(feature_types)
        cat_features = set(cat_features) if cat_features is not None else set()

        self.edges_ = []
        for j in range(n_features):
            if not self._should_quantize(j, feature_types, cat_features):
                self.edges_.append(None)
                continue
            self.edges_.append(self._fit_column(X[:, j], y))
        return self

    def _column_to_bins(self, col: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Нужна для задания: 2.6.1, 2.6.2."""
        col = self._column_as_float(col)
        out = np.zeros(col.shape[0], dtype=float)
        finite = np.isfinite(col)
        if not finite.any():
            return out

        clipped = np.clip(col[finite], edges[0], edges[-1])
        inner = edges[1:-1] if edges.size > 2 else edges[:1]
        bins = np.searchsorted(inner, clipped, side="right")
        bins = np.clip(bins, 0, len(edges) - 2)
        out[finite] = bins.astype(float)
        return out

    @staticmethod
    def _column_as_float(col: np.ndarray) -> np.ndarray:
        """Нужна для задания: 2.6."""
        col = np.asarray(col)
        if col.dtype.kind in "iuuf":
            return col.astype(float, copy=False)
        codes, _ = pd.factorize(col, sort=False)
        return codes.astype(float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Нужна для задания: 2.6.1, 2.6.2."""
        n_samples, n_features = X.shape
        X_out = np.empty((n_samples, n_features), dtype=float)
        for j, edges in enumerate(self.edges_):
            col = X[:, j]
            if edges is None:
                X_out[:, j] = self._column_as_float(col)
            else:
                X_out[:, j] = self._column_to_bins(col, edges)
        return X_out


class BoostingClassifier(ClassifierMixin):
    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = True,
        # --- §1.2 ---
        early_stopping_rounds: int | None = 0,
        eval_metric: str | None = None,
        use_best_model: bool = False,
        # --- §2.2 / §2.3 ---
        cat_features: Iterable | None = None,
        # --- §2.4 ---
        feature_types: Iterable[str] | None = None,
        l2: float = 0.0,
        # --- §2.5.1 ---
        subsample: float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str | None = "Bernoulli",
        # --- §2.5.2 ---
        rsm: float = 1.0,
        # --- §2.5.3 ---
        goss: bool = False,
        goss_k: float = 0.2,
        # --- §2.6 ---
        quantization_type: str | None = None,
        nbins: int = 255,
        # --- §2.9 ---
        dart: bool = False,
        dropout_rate: float = 0.05,
        # --- §2.11 ---
        loss: str = "BCE",
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = [0] * (n_estimators)
        self.gammas = [0] * (n_estimators)

        self.random_state = random_state  # не забудьте вставить его везде, где у вас возникает рандом
        self.verbose = verbose

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.grad_fn = lambda y, z: -y / (1 + np.exp(y * z))  
        self.hess_fn = lambda y, z: self.sigmoid(y * z) * (1 - self.sigmoid(y * z))  # §2.4

        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.use_best_model = use_best_model
        self.cat_features = cat_features
        self.feature_types = feature_types
        self.l2 = l2
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.goss = goss
        self.goss_k = goss_k
        self.quantization_type = quantization_type
        self.nbins = nbins
        self._quantizer: FeatureQuantizer | None = None
        self.dart = dart
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.focal_gamma = focal_gamma
        self.tree_weights: list[float] = []

        if self.loss == "Focal":
            g = self.focal_gamma

            def loss_fn(y, z):
                p = self.sigmoid(y * z)
                p = np.clip(p, 1e-15, 1 - 1e-15)
                return (-((1 - p) ** g) * np.log(p)).mean()

            def grad_fn(y, z):
                p = self.sigmoid(y * z)
                p = np.clip(p, 1e-15, 1 - 1e-15)
                dL_dp = -(-g * (1 - p) ** (g - 1) * np.log(p) + (1 - p) ** g / p)
                dp_dz = y * p * (1 - p)
                return dL_dp * dp_dz

            def hess_fn(y, z):
                p = self.sigmoid(y * z)
                return np.clip(p * (1 - p), 1e-6, None)

            self.loss_fn = loss_fn
            self.grad_fn = grad_fn
            self.hess_fn = hess_fn

        self._is_sklearn_tree = base_model_class.__name__ == "DecisionTreeRegressor"
        self._rsm_indices: list[np.ndarray] = []
        self._n_features: int | None = None

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Нужна для задания: 1, 1.2, 2.4, 2.5.1, 2.5.2, 2.5.3, 2.9."""
        pred_for_grad = self.train_predictions
        dart_scale = 1.0
        if self.dart and self.iteration > 0:
            rng = self._rng()
            drop = rng.random(self.iteration) < self.dropout_rate
            k = int(drop.sum())
            pred_for_grad = np.zeros_like(self.train_predictions)
            for j in range(self.iteration):
                if not self.models[j]:
                    continue
                feat_idx = self._rsm_indices[j]
                contrib = (
                    self.models[j].predict(X[:, feat_idx])
                    * self.gammas[j]
                    * self.learning_rate
                    * self.tree_weights[j]
                )
                if drop[j] and k > 0:
                    self.tree_weights[j] *= k / (k + 1)
                elif not drop[j]:
                    pred_for_grad += contrib
            if k > 0:
                dart_scale = 1.0 / k

        step = -self.grad_fn(y, pred_for_grad)
        hess = self.hess_fn(y, pred_for_grad)
        hess = np.clip(hess, 1e-6, None)

        feat_idx = self._feature_indices(X.shape[1])
        self._rsm_indices.append(feat_idx)
        X_tr = X[:, feat_idx]
        X_val_sub = X_val[:, feat_idx] if X_val is not None else None

        sample_weights = np.ones(len(step))
        if self.goss:
            sample_mask, step, hess = self._goss_mask_and_scale(step, hess)
        else:
            sample_mask, sample_weights = self._bootstrap_mask_and_weights(len(step))

        X_fit = X_tr[sample_mask]
        step_fit = step[sample_mask]
        hess_fit = hess[sample_mask]
        sw_fit = sample_weights[sample_mask]

        if self._is_sklearn_tree:
            curr_model = self.base_model_class(random_state=self.random_state, **self.base_model_params)
            if np.all(sw_fit == 1.0):
                curr_model.fit(X_fit, step_fit)
            else:
                curr_model.fit(X_fit, step_fit, sample_weight=sw_fit)
        else:
            step_fit = step_fit * sw_fit
            hess_fit = hess_fit * sw_fit
            tree_feature_types = (
                [self.feature_types[i] for i in feat_idx]
                if self.feature_types is not None
                else None
            )
            curr_model = self.base_model_class(
                feature_types=tree_feature_types,
                random_state=self.random_state,
                l2=self.l2,
                **self.base_model_params,
            )
            curr_model.fit(X_fit, step_fit, hess_fit)

        new_predictions_train = curr_model.predict(X_tr)

        if self.l2 != 0:
            opt_gamma = 1.0
        else:
            opt_gamma = self._find_optimal_gamma(y, self.train_predictions, new_predictions_train)

        opt_gamma *= dart_scale

        self.models[self.iteration] = curr_model
        self.gammas[self.iteration] = opt_gamma
        self.tree_weights.append(1.0)
        self.train_predictions += self.learning_rate * opt_gamma * new_predictions_train

        proba = self.sigmoid(self.train_predictions)
        loss_train = self.loss_fn(y, self.train_predictions)
        auc_score_train = roc_auc_score(y == 1, proba)
        self.history["train_loss"].append(loss_train)
        self.history["train_roc_auc"].append(auc_score_train)

        if X_val_sub is not None and y_val is not None:
            new_predictions_val = curr_model.predict(X_val_sub)
            self.val_predictions += self.learning_rate * opt_gamma * new_predictions_val
            proba_val = self.sigmoid(self.val_predictions)
            loss_val = self.loss_fn(y_val, self.val_predictions)
            auc_score_val = roc_auc_score(y_val == 1, proba_val)
            self.history["val_loss"].append(loss_val)
            self.history["val_roc_auc"].append(auc_score_val)

        self.iteration += 1

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Нужна для задания: 1, 1.2, 2.2, 2.3, 2.6."""
        if eval_set is None:
            X_valid, y_valid = X_train, y_train
        else:
            X_valid, y_valid = eval_set

        self.classes_ = np.unique(y_train)  # не рекомендуется убирать, нужно для калибровки
        self._n_features = X_train.shape[1]

        if self.cat_features is not None:
            self._cat_fit(X_train, y_train)
            X_train = self._cat_transform_ordered(X_train, y_train)
            X_valid = self._cat_transform(X_valid)

        y_enc = np.where(y_train == self.classes_[1], 1, -1)
        y_val_enc = np.where(y_valid == self.classes_[1], 1, -1)
        X_train = self._quantize_fit(X_train, y_enc)
        X_valid = self._quantize_transform(X_valid)

        self.train_predictions = np.zeros(X_train.shape[0])
        self.val_predictions = np.zeros(X_valid.shape[0])
        self.iteration = 0
        self._rsm_indices = []
        self.tree_weights = []

        y_train = y_enc
        y_valid = y_val_enc

        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)

        bad_change = 0
        best_score = -np.inf
        best_iteration = 0

        for i in estimator_range:
            self.partial_fit(X_train, y_train, X_valid, y_valid)

            if self.eval_metric and i > 0:
                current = self.history[self.eval_metric][-1]
                
                if "loss" in self.eval_metric:
                    is_better = current < best_score
                else:
                    is_better = current > best_score

                if is_better:
                    best_score = current
                    best_iteration = self.iteration
                    bad_change = 0
                else:
                    bad_change += 1

                if bad_change >= self.early_stopping_rounds:
                    if self.verbose:
                        print(f"\n Early Stopping на итерации {self.iteration}")
                    break

        if self.use_best_model and self.early_stopping_rounds > 0:
            self.models = self.models[:best_iteration]
            self.gammas = self.gammas[:best_iteration]
            self._rsm_indices = self._rsm_indices[:best_iteration]
            self.tree_weights = self.tree_weights[:best_iteration]
            for key in self.history:
                self.history[key] = self.history[key][:best_iteration]

        # чтобы было удобнее смотреть
        for key in self.history:
            self.history[key] = np.array(self.history[key])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Нужна для задания: 1, 2.2, 2.6."""
        if self.cat_features is not None and hasattr(self, "cat_mapa"):
            X = self._cat_transform(X)
        X = self._quantize_transform(X)
        proba = np.zeros(X.shape[0])
        for i in range(len(self.models)):
            if self.models[i] is not None and self.models[i] != 0:
                feat_idx = (
                    self._rsm_indices[i]
                    if i < len(self._rsm_indices)
                    else np.arange(X.shape[1])
                )
                w = self.tree_weights[i] if i < len(self.tree_weights) else 1.0
                proba += (
                    self.models[i].predict(X[:, feat_idx])
                    * self.gammas[i]
                    * self.learning_rate
                    * w
                )

        sigma_boy = self.sigmoid(proba)
        return np.column_stack([1 - sigma_boy, sigma_boy])

    def _find_optimal_gamma(
        self,
        y: np.ndarray,
        old_predictions: np.ndarray,
        new_predictions: np.ndarray,
    ) -> float:
        """Нужна для задания: 1."""
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Нужна для задания: 1."""
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])

    # вспомогательные методы §2 

    def _rng(self):
        """Нужна для задания: 2.5.1, 2.5.2, 2.5.3, 2.9."""
        seed = None if self.random_state is None else int(self.random_state) + self.iteration
        return np.random.default_rng(seed)

    def _goss_mask_and_scale(self, grad: np.ndarray, hess: np.ndarray):
        """Нужна для задания: 2.5.3."""
        n = len(grad)
        abs_grad = np.abs(grad)
        order = np.argsort(-abs_grad)
        n_top = max(1, int(np.ceil(self.goss_k * n)))
        top_idx = order[:n_top]
        rest_idx = order[n_top:]

        mask = np.zeros(n, dtype=bool)
        mask[top_idx] = True

        grad_fit = grad.copy()
        hess_fit = hess.copy()

        if len(rest_idx) > 0 and self.subsample < 1.0:
            rng = self._rng()
            sampled = rest_idx[rng.random(len(rest_idx)) < self.subsample]
            mask[sampled] = True
            scale = (1.0 - self.goss_k) / self.subsample
            grad_fit[sampled] *= scale
            hess_fit[sampled] *= scale
        else:
            mask[rest_idx] = True

        return mask, grad_fit, hess_fit

    def _bootstrap_mask_and_weights(self, n: int):
        """Нужна для задания: 2.5.1."""
        rng = self._rng()
        if self.bootstrap_type is None:
            return np.ones(n, dtype=bool), np.ones(n)

        if self.bootstrap_type == "Bernoulli":
            mask = rng.random(n) < self.subsample
            if not mask.any():
                mask[rng.integers(0, n)] = True
            return mask, np.ones(n)

        if self.bootstrap_type == "Bayesian":
            u = rng.random(n)
            u = np.clip(u, 1e-12, 1.0)
            weights = (-np.log(u)) ** self.bagging_temperature
            return np.ones(n, dtype=bool), weights


    def _cat_transform_ordered(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Нужна для задания: 2.3."""
        if self.cat_features is None:
            return X
        pos_class = self.classes_[1] if hasattr(self, "classes_") else 1
        y_bin = (y == pos_class).astype(float)
        X_out = X.copy()
        for feat in self.cat_features:
            col = X[:, feat]
            encoded = np.zeros(len(y))
            stats: dict = defaultdict(lambda: [0.0, 0.0])
            for idx in range(len(y)):
                key = col[idx]
                s, n = stats[key]
                encoded[idx] = s / n if n > 0 else 0.0
                stats[key][0] += y_bin[idx]
                stats[key][1] += 1.0
            X_out[:, feat] = encoded
        return X_out

    def _feature_indices(self, n_features: int) -> np.ndarray:
        """Нужна для задания: 2.5.2."""
        if self.rsm >= 1.0:
            return np.arange(n_features)
        rng = self._rng()
        k = max(1, int(np.round(self.rsm * n_features)))
        return np.sort(rng.choice(n_features, size=k, replace=False))

    def _cat_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Нужна для задания: 2.2."""
        if self.cat_features is None:
            return

        self.cat_mapa = defaultdict(dict)
        for i in self.cat_features:
            df = pd.DataFrame({"cat": X[:, i], "target": y})
            df["target_bin"] = df["target"] == self.classes_[1]
            encoding = df.groupby("cat")["target_bin"].mean().to_dict()
            self.cat_mapa[i] = encoding

    def _cat_transform(self, X: np.ndarray) -> np.ndarray:
        """Нужна для задания: 2.2."""
        if self.cat_features is None:
            return X

        X_transformed = X.copy()
        for i in self.cat_features:
            encoded = pd.Series(X[:, i]).map(self.cat_mapa[i]).fillna(0).values
            X_transformed[:, i] = encoded
        return X_transformed

    def _quantize_fit(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Нужна для задания: 2.6."""
        if self.quantization_type is None:
            self._quantizer = None
            return X
        self._quantizer = FeatureQuantizer(self.quantization_type, self.nbins)
        self._quantizer.fit(X, self.feature_types, self.cat_features, y=y)
        return self._quantizer.transform(X)

    def _quantize_transform(self, X: np.ndarray) -> np.ndarray:
        """Нужна для задания: 2.6."""
        if self._quantizer is None:
            return X
        return self._quantizer.transform(X)

    def plot_history(self, keys: str | Iterable[str]):
        """Нужна для задания: 1.2."""
        if isinstance(keys, str):
            keys = [keys]
        plt.figure(figsize=(12, 6))
        for key in keys:
            if key in self.history:
                plt.plot(self.history[key], label=key, linewidth=2)
        plt.xlabel("Итерация")
        plt.ylabel("Значение")
        plt.title("История метрик")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.show()

    def get_feature_importance(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        type: str = "split",
    ) -> np.ndarray:
        """Нужна для задания: 2.7, 2.8."""
        n_features = self._n_features or (X.shape[1] if X is not None else 0)
        importance = np.zeros(n_features)
        n_models = len(self.models)

        for i in range(n_models):
            model = self.models[i]
            if model in (0, None):
                continue
            weight = abs(self.gammas[i] * self.learning_rate * self.tree_weights[i])
            feat_idx = self._rsm_indices[i] if i < len(self._rsm_indices) else np.arange(n_features)

            if type == "split":
                if self._is_sklearn_tree and hasattr(model, "feature_importances_"):
                    local_imp = model.feature_importances_
                else:
                    local_imp = self._custom_tree_split_importance(model, len(feat_idx))
                for local_j, global_j in enumerate(feat_idx):
                    importance[global_j] += weight * local_imp[local_j]
            elif type == "gain":
                X_imp = X.copy()
                if self.cat_features is not None and hasattr(self, "cat_mapa"):
                    X_imp = self._cat_transform_ordered(X_imp, y)
                X_imp = self._quantize_transform(X_imp)
                y_int = np.where(y == self.classes_[1], 1, -1)
                grad = -self.grad_fn(y_int, np.zeros(X_imp.shape[0]))
                if self._is_sklearn_tree and hasattr(model, "tree_"):
                    local_imp = self._sklearn_tree_gain_importance(
                        model, X_imp[:, feat_idx], grad
                    )
                else:
                    local_imp = self._custom_tree_split_importance(model, len(feat_idx))
                for local_j, global_j in enumerate(feat_idx):
                    importance[global_j] += weight * local_imp[local_j]
            

        total = importance.sum()
        if total > 0:
            importance /= total
        return importance

    @staticmethod
    def _custom_tree_split_importance(model, n_local_features: int) -> np.ndarray:
        """Нужна для задания: 2.7."""
        imp = np.zeros(n_local_features)
        tree = getattr(model, "_tree", {})

        def walk(node):
            if node.get("type") != "nonterminal":
                return
            f = node["feature_split"]
            imp[f] += 1.0
            walk(node["left_child"])
            walk(node["right_child"])

        if tree:
            walk(tree)
        if imp.sum() > 0:
            imp /= imp.sum()
        return imp

    @staticmethod
    def _sklearn_tree_gain_importance(model, X, grad: np.ndarray) -> np.ndarray:
        """Нужна для задания: 2.8."""
        tree = model.tree_
        n_features = X.shape[1]
        importance = np.zeros(n_features)
        sample_idx = np.arange(len(X))
        if len(sample_idx) > 5000:
            sample_idx = np.random.default_rng(0).choice(sample_idx, 5000, replace=False)

        for idx in sample_idx:
            node_id = 0
            g = abs(grad[idx])
            while tree.children_left[node_id] != _tree.TREE_LEAF:
                feat = tree.feature[node_id]
                importance[feat] += g
                if X[idx, feat] <= tree.threshold[node_id]:
                    node_id = tree.children_left[node_id]
                else:
                    node_id = tree.children_right[node_id]

        if importance.sum() > 0:
            importance /= importance.sum()
        return importance


def find_best_split(feature_vector, grad_vector, hess_vector, lambda_reg=1.0):
    """Нужна для задания: 2.4.1."""
    indexes = np.argsort(feature_vector)
    x_sorted = feature_vector[indexes]
    g_sorted = grad_vector[indexes]
    h_sorted = hess_vector[indexes]

    mask = x_sorted[:-1] != x_sorted[1:]

    thresholds = (x_sorted[:-1][mask] + x_sorted[1:][mask]) / 2

    if len(thresholds) == 0:
        return thresholds, np.array([]), None, None

    G_left = np.cumsum(g_sorted)[:-1][mask]
    H_left = np.cumsum(h_sorted)[:-1][mask]

    G_total = g_sorted.sum()
    H_total = h_sorted.sum()

    G_right = G_total - G_left
    H_right = H_total - H_left

    scores = (G_left**2 / (H_left + lambda_reg) + G_right**2 / (H_right + lambda_reg) - G_total**2 / (H_total + lambda_reg))

    index = scores.argmax()

    return thresholds, scores, thresholds[index], scores[index]


class DecisionTree:
    """Нужна для задания: 2.4.2."""

    def __init__(self, feature_types, max_depth=None, min_samples_split=None,
                 min_samples_leaf=None, l2=1.0, random_state=None):
        """Нужна для задания: 2.4.2."""
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._l2 = l2
        self.random_state = random_state

    def _fit_node(self, sub_X, sub_grad, sub_hess, node, depth=0):
        """Нужна для задания: 2.4.2."""
        if len(sub_grad) == 0:
            node["type"] = "terminal"
            node["value"] = sub_grad.sum() / (sub_hess.sum() + self._l2)
            return
        
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["value"] = sub_grad.sum() / (sub_hess.sum() + self._l2)
            return
        
        if self._min_samples_split is not None and len(sub_grad) < self._min_samples_split:
            node["type"] = "terminal"
            node["value"] = sub_grad.sum() / (sub_hess.sum() + self._l2)
            return

        feature_best, threshold_best, score_best, split_best = None, None, -np.inf, None
        
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                unique_cats = np.unique(sub_X[:, feature])
                categories_map = {cat: i for i, cat in enumerate(unique_cats)}
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, score = find_best_split(feature_vector, sub_grad, sub_hess, self._l2)

            if threshold is None:
                continue

            current_split = feature_vector < threshold

            if self._min_samples_leaf is not None:
                if current_split.sum() < self._min_samples_leaf:
                    continue
                if (~current_split).sum() < self._min_samples_leaf:
                    continue

            if score > score_best:
                feature_best = feature
                score_best = score
                split_best = current_split

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [cat for cat, idx in categories_map.items() if idx < threshold]

        if feature_best is None:
            node["type"] = "terminal"
            node["value"] = sub_grad.sum() / (sub_hess.sum() + self._l2)
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}
        
        self._fit_node(sub_X[split_best], sub_grad[split_best], sub_hess[split_best], 
                       node["left_child"], depth + 1)
        self._fit_node(sub_X[~split_best], sub_grad[~split_best], sub_hess[~split_best], 
                       node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        """Нужна для задания: 2.4.2."""
        if node["type"] == "terminal":
            return node["value"]

        feature_best = node["feature_split"]

        if self._feature_types[feature_best] == "real":
            if x[feature_best] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:  
            if x[feature_best] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, grad, hess):
        """Нужна для задания: 2.4.2."""
        self._fit_node(X, grad, hess, self._tree, depth=0)

    def predict(self, X):
        """Нужна для задания: 2.4.2."""
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)