"""Advanced model collection for the NSL-KDD anomaly detection study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:  # XGBoost is optional during development
    from xgboost import XGBClassifier  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    LGBMClassifier = None  # type: ignore


@dataclass
class TrainingResult:
    """Container for tracking model training metadata."""

    model_name: str
    training_time: float
    status: str
    error: Optional[str] = None


class AdvancedModels:
    """Collection of advanced supervised models for NSL-KDD.

    The class mirrors the workflow of :class:`models.baseline.BaselineModels`
    to keep the training and evaluation pipeline consistent. Each model is
    instantiated with sensible defaults that work well on the NSL-KDD dataset
    without requiring heavy hyper-parameter tuning.
    """

    def __init__(self, random_state: int = 42, n_jobs: int = -1) -> None:
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models: Dict[str, BaseEstimator] = self._initialize_models()
        self.trained_models: Dict[str, BaseEstimator] = {}
        self.results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------
    def _initialize_models(self) -> Dict[str, BaseEstimator]:
        """Create all supported advanced models.

        Returns
        -------
        Dict[str, ClassifierMixin]
            Mapping of model names to scikit-learn compatible estimators.
        """

        models: Dict[str, BaseEstimator] = {}

        if XGBClassifier is not None:
            models["xgboost"] = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=400,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                reg_alpha=0.0,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                tree_method="hist",
                use_label_encoder=False,
            )
        else:
            print("⚠️ XGBoost not available. Install `xgboost` to enable this model.")

        if LGBMClassifier is not None:
            models["lightgbm"] = LGBMClassifier(
                objective="binary",
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                class_weight="balanced",
            )
        else:
            print("⚠️ LightGBM not available. Install `lightgbm` to enable this model.")

        models["gradient_boosting"] = GradientBoostingClassifier(
            random_state=self.random_state,
            learning_rate=0.1,
            n_estimators=300,
            max_depth=3,
            subsample=0.9,
        )

        models["extra_trees"] = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=False,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        # MLP benefits from scaling; we wrap it in a pipeline to avoid
        # assuming pre-scaled features.
        models["mlp"] = make_pipeline(
            StandardScaler(with_mean=False),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=120,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.random_state,
            ),
        )

        # Soft voting ensemble across tree, boosting, and neural network models.
        # The estimators are freshly initialised to avoid shared states.
        voting_estimators: List[tuple[str, BaseEstimator]] = []
        if "extra_trees" in models:
            voting_estimators.append(("extra_trees", models["extra_trees"]))
        if "gradient_boosting" in models:
            voting_estimators.append(("gradient_boosting", models["gradient_boosting"]))
        if "mlp" in models:
            voting_estimators.append(("mlp", models["mlp"]))

        # Only create the voting classifier when at least two base estimators are present.
        if len(voting_estimators) >= 2:
            voting_estimators_for_vc: List[tuple[str, BaseEstimator]] = []
            for name, estimator in voting_estimators:
                # Re-create estimators to prevent shared fitted state between
                # the standalone model and the voting ensemble.
                voting_estimators_for_vc.append((name, self._clone_estimator(estimator)))

            models["voting_classifier"] = VotingClassifier(
                estimators=voting_estimators_for_vc,
                voting="soft",
                n_jobs=self.n_jobs,
            )

        return models

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clone_estimator(estimator: BaseEstimator) -> BaseEstimator:
        """Return a fresh clone of an estimator.

        Parameters
        ----------
        estimator:
            Scikit-learn compatible estimator to clone. Pipelines are handled
            transparently via `sklearn.base.clone`.
        """

        from sklearn.base import clone

        return clone(estimator)

    @staticmethod
    def _is_probabilistic(model: BaseEstimator) -> bool:
        return hasattr(model, "predict_proba")

    # ------------------------------------------------------------------
    # Training routines
    # ------------------------------------------------------------------
    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> TrainingResult:
        """Train a single advanced model."""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Choices: {list(self.models)}")

        model = self.models[model_name]
        if verbose:
            print(f"🤖 Training advanced model: {model_name}")

        import time

        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            self.trained_models[model_name] = model
            status = "success"
            error: Optional[str] = None
            if verbose:
                print(f"✅ {model_name} trained in {training_time:.2f}s")
        except Exception as exc:  # pragma: no cover - runtime feedback only
            training_time = time.time() - start_time
            status = "failed"
            error = str(exc)
            if verbose:
                print(f"❌ Failed to train {model_name}: {error}")

        return TrainingResult(model_name, training_time, status, error)

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, TrainingResult]:
        """Train multiple advanced models in sequence."""

        if include is None:
            candidates: List[str] = list(self.models.keys())
        else:
            candidates = [name for name in include if name in self.models]

        exclude_set = set(exclude or [])
        to_train = [name for name in candidates if name not in exclude_set]

        if verbose:
            print(f"🚀 Training {len(to_train)} advanced models")

        results: Dict[str, TrainingResult] = {}
        for model_name in to_train:
            results[model_name] = self.train_model(model_name, X_train, y_train, verbose=verbose)

        return results

    # ------------------------------------------------------------------
    # Evaluation routines
    # ------------------------------------------------------------------
    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        average: str = "weighted",
        dataset: str = "test",
    ) -> Dict[str, Any]:
        """Evaluate a single trained advanced model."""

        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained yet")

        model = self.trained_models[model_name]

        import time

        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average, zero_division=0)

        roc_auc: Optional[float] = None
        try:
            if self._is_probabilistic(model):
                y_proba = model.predict_proba(X_test)
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average=average)
        except Exception:
            roc_auc = None

        result = {
            "model_name": model_name,
            "dataset": dataset,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "prediction_time": prediction_time,
        }

        return result

    def evaluate_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        average: str = "weighted",
        sort_by: str = "f1_score",
        verbose: bool = True,
        dataset: str = "validation",
    ) -> pd.DataFrame:
        """Evaluate every trained advanced model."""

        if verbose:
            print(f"📊 Evaluating {len(self.trained_models)} advanced models")

        self.results = []
        for model_name in self.trained_models:
            if verbose:
                print(f"�� Evaluating {model_name}")
            metrics = self.evaluate_model(
                model_name,
                X_test,
                y_test,
                average=average,
                dataset=dataset,
            )
            self.results.append(metrics)
            if verbose:
                print(
                    f"   F1={metrics['f1_score']:.3f} | "
                    f"Acc={metrics['accuracy']:.3f} | Prec={metrics['precision']:.3f} | "
                    f"Rec={metrics['recall']:.3f}"
                )

        results_df = pd.DataFrame(self.results)
        if not results_df.empty and sort_by in results_df.columns:
            results_df = results_df.sort_values(sort_by, ascending=False)

        return results_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_models(self, output_dir: str | Path, results_filename: str = "advanced_results.csv") -> None:
        """Persist trained models and aggregated results to disk."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        import joblib

        for model_name, model in self.trained_models.items():
            model_path = output_path / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            print(f"💾 Saved {model_name} to {model_path}")

        if self.results:
            results_df = pd.DataFrame(self.results)
            results_path = output_path / results_filename
            results_df.to_csv(results_path, index=False)
            print(f"💾 Saved evaluation results to {results_path}")

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------
    def get_best_model(self, metric: str = "f1_score") -> tuple[str, BaseEstimator]:
        if not self.results:
            raise ValueError("No evaluation results available. Run `evaluate_all` first.")

        results_df = pd.DataFrame(self.results)
        best_row = results_df.loc[results_df[metric].idxmax()]
        model_name = best_row["model_name"]
        return model_name, self.trained_models[model_name]


__all__ = ["AdvancedModels", "TrainingResult"]
