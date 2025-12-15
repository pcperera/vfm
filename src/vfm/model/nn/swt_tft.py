import pandas as pd
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import MultiNormalizer, GroupNormalizer
from pytorch_forecasting.metrics import MultiLoss, QuantileLoss

NUM_WORKERS = 4

class SWTTFTModel:

    def __init__(self, df: pd.DataFrame,
                 dependent_variables: list[str],
                 independent_variables: list[str]):

        # ----------------------------
        # 1. USE ONLY REAL DATA FOR TRAIN/VAL/TEST
        # ----------------------------
        self.df_real = df.dropna(subset=dependent_variables).copy()
        self.df_real["is_real"] = 1

        # (Optional) If your main DF contains hybrid model reconstructions,
        # you can mark them as reconstructed.
        df["is_real"] = df[dependent_variables].notna().all(axis=1).astype(int)

        # Train on ALL data (real + reconstructed)
        self.df_all = df.copy()

        self._batch_size = 4
        min_encoder_length = 1
        max_encoder_length = 5
        min_prediction_length = 1
        max_prediction_length = 5

        # ----------------------------
        # 2. TRAINING DATASET (real + reconstructed)
        # ----------------------------
        self.training_dataset = TimeSeriesDataSet(
            self.df_all,
            time_idx="time_idx",
            target=dependent_variables,
            group_ids=["well_id"],
            min_encoder_length=min_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=min_prediction_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=["well_id"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=independent_variables,
            target_normalizer=MultiNormalizer(
                [GroupNormalizer(groups=["well_id"], transformation="softplus")
                 for _ in dependent_variables]
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

        self.train_data_loader = self.training_dataset.to_dataloader(
            train=True, batch_size=self._batch_size, num_workers=NUM_WORKERS, persistent_workers=True,
        )

        quantiles = [0.1, 0.5, 0.9]

        # ----------------------------
        # 3. DEFINE TFT
        # ----------------------------
        self.tft = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=3e-4,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.3,
            hidden_continuous_size=16,
            output_size=[len(quantiles) for _ in dependent_variables],
            loss=MultiLoss([QuantileLoss(quantiles=quantiles) for _ in dependent_variables]),
            log_interval=10,
            reduce_on_plateau_patience=5,
        )

        logger = TensorBoardLogger("lightning_logs")
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")
        lr_monitor = LearningRateMonitor()

        self.trainer = Trainer(
            max_epochs=2,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            logger=logger,
            callbacks=[early_stop_callback, lr_monitor],
        )

        # ----------------------------
        # 4. VALIDATION = REAL DATA ONLY
        # ----------------------------
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            self.df_real,
            predict=False,
            stop_randomization=True,
        )

        self.validation_data_loader = self.validation_dataset.to_dataloader(
            train=False,
            batch_size=self._batch_size,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )

        # ----------------------------
        # 5. TEST SET = STRICTLY REAL DATA ONLY
        # ----------------------------
        self.test_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            self.df_real,
            predict=False,
            stop_randomization=True,
        )

        self.test_dataloader = self.test_dataset.to_dataloader(
            train=False, batch_size=self._batch_size, num_workers=NUM_WORKERS, persistent_workers=True,
        )

        self.dependent_variables = dependent_variables

    # ================================================================
    # TRAINING
    # ================================================================
    def train(self):
        self.trainer.fit(
            model=self.tft,
            train_dataloaders=self.train_data_loader,
            val_dataloaders=self.validation_data_loader
        )

    # ================================================================
    # METRIC CALCULATION
    # ================================================================
    @staticmethod
    def compute_metrics(y_true, y_pred):
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    def evaluate_on_real_data(self):

        out = self.tft.predict(
            self.test_dataloader,
            mode="raw",
            return_x=True
        )

        # ------------------------------------------------------
        # Modern Prediction format: Prediction(output=Output(...))
        # ------------------------------------------------------
        if hasattr(out, "output"):
            # list[Tensor]: shape [batch, time, target, quantile]
            raw_pred_list = out.output.prediction
            target_list = out.output.target

            # concatenate over batches
            raw_pred = torch.cat(raw_pred_list, dim=0)
            y_true = torch.cat(target_list, dim=0)

        else:
            raise ValueError("Unexpected output format.")

        # ------------------------------------------------------
        # Extract median quantile (0.5 → index 1)
        # raw_pred: [N, pred_len, n_targets, n_quantiles]
        # ------------------------------------------------------
        median_index = 1
        preds = raw_pred[:, :, :, median_index].detach().cpu().numpy()
        preds = preds.reshape(-1, preds.shape[-1])

        y_true = y_true.detach().cpu().numpy()
        y_true = y_true.reshape(-1, y_true.shape[-1])

        # ------------------------------------------------------
        # Compute metrics
        # ------------------------------------------------------
        results = {}
        for i, target in enumerate(self.dependent_variables):
            results[target] = self.compute_metrics(y_true[:, i], preds[:, i])

        return results
