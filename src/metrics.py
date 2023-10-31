from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt


def generate_scores(y_test: pd.DataFrame, y_predicted: pd.DataFrame) -> None:
    # Loop over oil and gas flow rates
    for test_column in y_test.columns:
        # Calculate mean absolute percentage error (MAPE) and print score
        mape_score = round(mean_absolute_percentage_error(y_test[test_column], y_predicted[test_column]), 3) * 100
        mae_score = round(mean_absolute_error(y_test[test_column], y_predicted[test_column]), 3)
        print(f'{test_column} MAPE {mape_score}%, MAE {mae_score}')


def generate_plot(y_test: pd.DataFrame, y_predicted: pd.DataFrame) -> None:
    # Loop over oil and gas flow rates
    for test_column in y_test.columns:
        plt.figure(figsize=(25, 6))
        plt.scatter(y_test.index, y_test[test_column], label='Actual')
        plt.scatter(y_test.index, y_predicted[test_column], label='Predicted')
        plt.xlabel('Datetime')
        plt.ylabel('Flow Rate')
        plt.title(f'{test_column} Actual vs Predicted')
        plt.legend()
        plt.show()
