import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, MultiLoss
from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


class TemporalFusionTransformerModel:
    def __init__(self, train_data: pd.DataFrame, max_encoder_length: int = 4, max_prediction_length: int = 2, batch_size: int = 10, num_workers: int = 16):
        self.__train_data = train_data
        self.__max_encoder_length = max_encoder_length
        self.__max_prediction_length = max_prediction_length
        self.__batch_size = batch_size
        self.__num_workers = num_workers
        self.__accelerator = "gpu" if torch.cuda.is_available() else "cpu"

        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA version: {torch.cuda.get_device_name(0)}")

    def train(self, time_idx: str, time_series_idx: [], target_fields: [str], time_varying_known_reals: [str]):

        normalizers = []

        for idx in range(len(target_fields)):
            normalizers.append(GroupNormalizer(groups=time_series_idx, transformation="softplus"))

        # Training data set timeseries
        training_data_timeseries = TimeSeriesDataSet(
            self.__train_data,
            time_idx=time_idx,
            group_ids=time_series_idx,
            target=target_fields if len(target_fields) < 0 else target_fields[0],
            min_encoder_length=self.__max_encoder_length // 2,
            max_encoder_length=self.__max_encoder_length,
            min_prediction_length=self.__max_prediction_length // 2,
            max_prediction_length=self.__max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=target_fields,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            # target_normalizer=MultiNormalizer(normalizers=normalizers),
            allow_missing_timesteps=True,
        )

        # Validation data set timeseries
        validation_data_timeseries = TimeSeriesDataSet.from_dataset(training_data_timeseries, self.__train_data, predict=True, stop_randomization=True)

        # Create dataloaders for our model
        train_dataloader = training_data_timeseries.to_dataloader(train=True, batch_size=self.__batch_size, num_workers=self.__num_workers, pin_memory=True)
        val_dataloader = validation_data_timeseries.to_dataloader(train=False, batch_size=self.__batch_size, num_workers=self.__num_workers, pin_memory=True)

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
        lr_logger = LearningRateMonitor()
        logger = TensorBoardLogger("lightning_logs")

        model_checkpoints_directory = "checkpoints"

        # Create study
        study = optimize_hyperparameters(
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            model_path=model_checkpoints_directory,
            n_trials=1,
            max_epochs=1,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(64, 196),
            hidden_continuous_size_range=(64, 196),
            attention_head_size_range=(2, 4),
            learning_rate_range=(0.0001, 0.01),
            dropout_range=(0.0, 0.3),
            use_learning_rate_finder=False,  # Use Optuna to find ideal learning rate or use in-built learning rate finder
            trainer_kwargs=dict(accelerator=self.__accelerator, devices=1, callbacks=[lr_logger, early_stop_callback]),
            verbose=2,
            loss=QuantileLoss(),
            reduce_on_plateau_patience=4)

        # Get best hyperparameters
        best_trial = study.best_trial
        best_hyperparams = best_trial.params
        print(f"Best hyper-parameters : {best_hyperparams}")

        # with open(f"hyperparameters.yaml", "w") as f:
        #     yaml.dump(best_hyperparams, f)

        best_tft = TemporalFusionTransformer.from_dataset(
            training_data_timeseries,
            learning_rate=best_hyperparams["learning_rate"],
            hidden_size=best_hyperparams["hidden_size"],
            attention_head_size=best_hyperparams["attention_head_size"],
            dropout=best_hyperparams["dropout"],
            hidden_continuous_size=best_hyperparams["hidden_continuous_size"],
            output_size=7,  # There are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
            log_interval=10,
            reduce_on_plateau_patience=4)

        if torch.cuda.is_available():
            best_tft = best_tft.cuda()

        # print(f"Best trial check point path {checkpoint_file_path}.")
        # best_tft = best_tft.load_from_checkpoint(checkpoint_file_path)

        actual_values = []

        for idx in range(len(target_fields)):
            print(f"INDEX {idx}")
            actual_values.append(torch.cat([y[0][idx] for x, y in iter(val_dataloader)]))
            if torch.cuda.is_available():
                actual_values[idx] = actual_values[idx].cuda()

        # for x, y in iter(val_dataloader):
        #     print(x)
        #     print(y)

        predictions = best_tft.predict(val_dataloader)

        # for idx in range(len(target_fields)):
        #     # Average p50 loss overall
        #     print(f"{target_fields[idx]} Median Loss Overall = {(actual_values[idx] - predictions[idx]).abs().mean().item()}")
        #     # Average p50 loss per time series
        #     print((actual_values[idx] - predictions[idx]).abs().mean(axis=1))

        # i = 0
        # for prediction in predictions:
        #     print(f"Prediction {i} = {prediction}");
        #     i += 1

        raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)

        return raw_predictions, best_tft
