
import pandas as pd
import torch
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import MultiLoss, QuantileLoss
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import MultiNormalizer, GroupNormalizer
from pytorch_forecasting.metrics import MultiLoss, QuantileLoss


class SWTTFTModel:

    def __init__(self, df: pd.DataFrame, dependent_variables: list[str] = None, independent_variables: list[str] = None):
        self._df_train = df.dropna(subset=dependent_variables).copy()
        self._batch_size = 4
        max_encoder_length = 1  
        min_encoder_length = 1  
        max_prediction_length = 1
        min_prediction_length = 1


        self.training_dataset = TimeSeriesDataSet(
            self._df_train,
            time_idx="time_idx",
            target=dependent_variables,
            group_ids=["well_id"],
            min_encoder_length=min_encoder_length,  # encoder length
            max_encoder_length=max_encoder_length,
            min_prediction_length=min_prediction_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=["well_id"],
            time_varying_known_reals=["time_idx"],  # we only know timestamp
            time_varying_unknown_reals=independent_variables,
            target_normalizer=MultiNormalizer(
                [GroupNormalizer(groups=["well_id"], transformation="softplus") for _ in dependent_variables]
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

        # Create train data loader
        self.train_data_loader = self.training_dataset.to_dataloader(train=True, batch_size=self._batch_size, num_workers=0)
        quantiles = [0.1, 0.5, 0.9]

        # Define TFT model
        self.tft = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=3e-4,
            hidden_size=32,  # smaller due to small training data
            attention_head_size=4,
            dropout=0.3,
            hidden_continuous_size=16,
            output_size=[len(quantiles) for _ in dependent_variables],  # number of targets
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

        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            self._df_train,               # same dataframe unless you have a separate val set
            predict=False,
            stop_randomization=True,
        )

        self.validation_data_loader = self.validation_dataset.to_dataloader(
            train=False,
            batch_size=self._batch_size,
            num_workers=0
        )

    def train(self):
        print(type(self.tft))

        self.trainer.fit(
            model=self.tft,
            train_dataloaders=self.train_data_loader,
            val_dataloaders=self.validation_data_loader
        )

    def predict(self):
        # Use the last sequence for prediction
        raw_predictions, x = self.tft.predict(self.train_data_loader, mode="raw", return_x=True)
        predictions = self.tft.predict(self.train_data_loader)
        return predictions[:5]


    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        print("val_loss", loss)
        return loss
