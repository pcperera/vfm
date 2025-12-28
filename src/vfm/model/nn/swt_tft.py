# ============================================================
# physics_informed_tft.py
# ============================================================

import numpy as np
import pandas as pd
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import QuantileLoss


NUM_WORKERS = 4


# ============================================================
# METRICS
# ============================================================

def regression_metrics(y_true, y_pred):
    eps = 1e-8
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r2 = 1.0 - np.sum((y_true - y_pred) ** 2) / (
        np.sum((y_true - y_true.mean()) ** 2) + eps
    )
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0
    mpe = np.mean((y_pred - y_true) / (y_true + eps)) * 100.0

    return {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape (%)": float(mape),
        "mpe (%)": float(mpe),
    }


# ============================================================
# MODEL
# ============================================================

class PhysicsInformedTFT:
    """
    Physics-informed TFT for sparse MPFM supervision.

    Key principles:
    - Train ONLY on observed (MPFM) rows
    - Use FULL dataframe for encoder context
    - Validate/Test via predict_idx (never subset dataframe)
    - Multi-well via group_ids
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dependent_vars: list[str],
        independent_vars: list[str],
        encoder_length: int = 16,
        prediction_length: int = 1,
        batch_size: int = 8,
    ):

        # ----------------------------------------------------
        # 0. Defensive preprocessing
        # ----------------------------------------------------
        self.df = df.copy()

        # enforce ordering + unique index (critical)
        self.df = (
            self.df
            .sort_values(["well_id", "time_idx"])
            .reset_index(drop=True)
        )

        # observed rows = ground truth
        self.obs_mask = self.df[dependent_vars].notna().all(axis=1)
        self.predict_idx = self.df.index[self.obs_mask]

        self.dependent_vars = dependent_vars
        self.batch_size = batch_size

        # ----------------------------------------------------
        # 1. TRAINING DATASET (OBSERVED ROWS ONLY)
        # ----------------------------------------------------
        df_train = self.df[self.obs_mask].copy()

        self.training_dataset = TimeSeriesDataSet(
            df_train,
            time_idx="time_idx",
            target=dependent_vars,
            group_ids=["well_id"],

            min_encoder_length=encoder_length,
            max_encoder_length=encoder_length,
            min_prediction_length=prediction_length,
            max_prediction_length=prediction_length,

            static_categoricals=["well_id"],
            time_varying_known_reals=independent_vars,
            time_varying_unknown_reals=[],

            target_normalizer=MultiNormalizer(
                [
                    GroupNormalizer(
                        groups=["well_id"],
                        transformation="softplus",
                    )
                    for _ in dependent_vars
                ]
            ),

            allow_missing_timesteps=True,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        self.train_loader = self.training_dataset.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )

        # ----------------------------------------------------
        # 2. VALIDATION DATASET (FULL DF + predict_idx)
        # ----------------------------------------------------
        self.val_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            self.df,
            predict=True,
            stop_randomization=True,
            predict_idx=self.predict_idx,
        )

        self.val_loader = self.val_dataset.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )

        # ----------------------------------------------------
        # 3. TEST DATASET (same structure; split predict_idx if needed)
        # ----------------------------------------------------
        self.test_dataset = self.val_dataset
        self.test_loader = self.val_loader

        # ----------------------------------------------------
        # 4. TFT MODEL
        # ----------------------------------------------------
        quantiles = [0.1, 0.5, 0.9]

        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            hidden_size=32,
            attention_head_size=4,
            hidden_continuous_size=16,
            dropout=0.2,
            learning_rate=3e-4,
            output_size=[len(quantiles) for _ in dependent_vars],
            loss=QuantileLoss(quantiles=quantiles),
            log_interval=50,
            reduce_on_plateau_patience=5,
        )

        self.trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed" if torch.cuda.is_available() else 32,
            max_epochs=30,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=6),
                LearningRateMonitor(),
            ],
            logger=TensorBoardLogger("tft_logs"),
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    def train(self):
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )

    # --------------------------------------------------------
    # EVALUATE (OBSERVED ROWS ONLY)
    # --------------------------------------------------------
    def evaluate(self):

        out = self.model.predict(
            self.test_loader,
            mode="raw",
            return_x=True,
        )

        raw_pred = torch.cat(out.output.prediction, dim=0)
        y_true = torch.cat(out.output.target, dim=0)

        # median quantile (0.5)
        preds = raw_pred[..., 1].detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        preds = preds.reshape(-1, preds.shape[-1])
        y_true = y_true.reshape(-1, y_true.shape[-1])

        results = {}
        for i, target in enumerate(self.dependent_vars):
            results[target] = regression_metrics(
                y_true[:, i],
                preds[:, i],
            )

        return results
