from abc import ABC, abstractmethod
from src.vfm.constants import *
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class BasePhysicsInformedHybridModel(ABC):

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """
        Fit the hybrid model using the provided DataFrame.

        Parameters:
            df (pd.DataFrame): Input data for fitting the model.
        """
        pass