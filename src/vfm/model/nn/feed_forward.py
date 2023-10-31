import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from src.metrics import generate_scores, generate_plot
from pandas import DataFrame


class FeedForwardNeuralNetworkModel(pl.LightningModule):
    def __init__(self, x_train: DataFrame, y_train: DataFrame, x_test: DataFrame, y_test: DataFrame, num_hidden_layers=5, num_hidden_features=10, learning_rate=0.001):
        super(FeedForwardNeuralNetworkModel, self).__init__()
        self.__x_train = x_train
        self.__y_train = y_train
        self.__x_test = x_test
        self.__y_test = y_test
        self.__y_predicted = []
        self.__batch_size = 64
        self.__num_workers = 4

        # Hyperparameters
        self.__num_hidden_layers = num_hidden_layers
        self.__num_hidden_features = num_hidden_features
        self.__learning_rate = learning_rate

        # Loss function
        self.__criterion = nn.MSELoss()

        self.__fc1 = nn.Linear(self.__x_train.shape[1], num_hidden_features)  # Input layer

        # Hidden layers
        self.__hidden_layers = nn.ModuleList([
            nn.Linear(num_hidden_features, num_hidden_features)
            for _ in range(num_hidden_layers)
        ])

        self.__fc3 = nn.Linear(num_hidden_features, self.__y_train.shape[1])  # Output layer

    def forward(self, x):
        out = self.__fc1(x)
        for hidden_layer in self.__hidden_layers:
            out = torch.relu(hidden_layer(out))
        out = self.__fc3(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.__criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.__criterion(y_hat, y)
        self.log('test_loss', loss)
        # Iterate over individual samples and append predictions
        for i in range(len(y)):
            self.__y_predicted.append(y_hat[i])
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.__learning_rate)

    def train_dataloader(self):
        train_dataset = TensorDataset(torch.Tensor(self.__x_train.values), torch.Tensor(self.__y_train.values))
        return DataLoader(train_dataset, batch_size=self.__batch_size, num_workers=self.__num_workers, shuffle=False)

    def test_dataloader(self):
        self.__y_predicted.clear()
        train_dataset = TensorDataset(torch.Tensor(self.__x_test.values), torch.Tensor(self.__y_test.values))
        return DataLoader(train_dataset, batch_size=self.__batch_size, num_workers=self.__num_workers, shuffle=False)

    def get_predicted(self):
        return self.__y_predicted


class FeedForwardNeuralNetworkEvaluator:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.__x_train = x_train
        self.__y_train = y_train
        self.__x_test = x_test
        self.__y_test = y_test
        self.__y_predicted = None
        self.__study = optuna.create_study(direction="minimize")
        self.__best_model: FeedForwardNeuralNetworkModel = None

        # Create a PyTorch Lightning Trainer
        self.__trainer = None
        self.__checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename="best_model", monitor="train_loss", mode="min")

    def train(self):
        self.__study.optimize(self.__objective, n_trials=5)

    def test(self):
        best_trial = self.__study.best_trial
        best_model_path = best_trial.user_attrs["best_model_path"]

        print('Best model :')
        best_num_hidden_layers = best_trial.params["num_hidden_layers"]
        best_num_hidden_features = best_trial.params["num_hidden_features"]
        best_learning_rate = best_trial.params["learning_rate"]
        best_num_epochs = best_trial.params["num_epochs"]
        print(f'num_hidden_layers = {best_num_hidden_layers}, num_hidden_features = {best_num_hidden_features}, '
              f'learning_rate = {best_learning_rate}, num_epochs = {best_num_epochs}')

        # Load the best model checkpoint into memory
        self.__best_model = FeedForwardNeuralNetworkModel.load_from_checkpoint(best_model_path,
                                                                               x_train=self.__x_train,
                                                                               y_train=self.__y_train,
                                                                               x_test=self.__x_test,
                                                                               y_test=self.__y_test,
                                                                               num_hidden_layers=best_num_hidden_layers,
                                                                               num_hidden_features=best_num_hidden_features,
                                                                               learning_rate=best_learning_rate)

        test_results = self.__trainer.test(model=self.__best_model)
        print(f"Test loss {test_results[0]['test_loss']}")

        # Convert tensors to NumPy arrays
        numpy_arrays = [tensor.numpy() for tensor in self.__best_model.get_predicted()]

        print('Predictions:')
        for predicted_column in numpy_arrays:
            print(f'q_oil: {predicted_column[0].item()}, q_gas: {predicted_column[1].item()},  q_water: {predicted_column[2].item()}')

        # Create the DataFrame
        self.__y_predicted = DataFrame(numpy_arrays, columns=[self.__y_test.columns])

    def results(self):
        generate_scores(y_test=self.__y_test, y_predicted=self.__y_predicted)
        generate_plot(y_test=self.__y_test, y_predicted=self.__y_predicted)

    def __objective(self, trial):
        # Define the hyperparameters to be tuned
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 10, 20)
        num_hidden_features = trial.suggest_int("num_hidden_features", 16, 64)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        num_epochs = trial.suggest_int("num_epochs", 100, 150)

        print(f'num_hidden_layers = {num_hidden_layers}, num_hidden_features = {num_hidden_features},'
              f' learning_rate = {learning_rate}, num_epochs = {num_epochs}')

        model = FeedForwardNeuralNetworkModel(self.__x_train, self.__y_train, self.__x_test, self.__y_test,
                                              num_hidden_layers=num_hidden_layers,
                                              num_hidden_features=num_hidden_features,
                                              learning_rate=learning_rate)
        self.__trainer = pl.Trainer(max_epochs=num_epochs,
                                    callbacks=[EarlyStopping(monitor='train_loss'), self.__checkpoint_callback])

        # Train the model
        self.__trainer.fit(model)

        # Save the best checkpoint path in user attributes
        trial.set_user_attr("best_model_path", self.__checkpoint_callback.best_model_path)

        # Return the validation loss as the objective value to be minimized
        return self.__trainer.callback_metrics["train_loss"].item()
