# Stock Price Prediction AI Project

## Introduction
This project aims to predict stock prices using various deep learning models such as Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), Bidirectional LSTM (BiLSTM), and Gated Recurrent Unit (GRU). The project also includes preprocessing steps to prepare the data for training and testing. Additionally, visualization techniques are employed to analyze the performance of the models.

## Dataset
The dataset used for this project consists of historical stock price data obtained from kaggle (https://www.kaggle.com/datasets/iamsouravbanerjee/nifty50-stocks-dataset/data). It includes features such as opening price, closing price, highest price, lowest price, and volume traded for each day.
This dataset looks like this:

This is pairplot.

![2](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/425ba2bd-dde7-420b-8bf8-a9d927c711b6)

This graph tells about the companies, with respect to their volumes

![3](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/58226c6c-864c-430f-b2f4-2d7136251f34)

This graph tells about the companies by change in percentage.

![4](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/1f7a3ba5-615d-4c07-8b2a-f51fc1363de7)

This graph tells about the Turn-Over Volume wise.

![5](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/7521a183-b3bc-4599-8a59-dcef222b5518)


This is HeatMap, which will tell us about the correlation of different attributes

![1](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/9a845786-963f-43a4-8b09-add7fc49a9c0)

According to this HeatMap, Open, high, low and LTP are highly co-related features, we can use any one of them to create model.

## Preprocessing
Before feeding the data into the deep learning models, several preprocessing steps are performed:
- **Normalization**: All features are normalized to ensure that they are on the same scale, preventing any particular feature from dominating the others.
- - **Noise and Volatility**:<br>
To address the noise and volatility in the data:
<br><br>
Data Preprocessing:<br><br> The data was preprocessed to remove commas and convert all columns to numeric values. Handling missing values by filling them with the mean of the column helps in reducing noise and ensuring that the dataset is clean before training the models.
<br><br>
Model Architecture:<br><br> Deep learning models like GRU, LSTM and BiLSTM are capable of capturing complex patterns and long-term dependencies in the data, which can help in filtering out noise and identifying relevant signals amidst volatility.
- - **Non-stationary Data**:<br>
To handle non-stationary data, the following preprocessing steps were implemented:
<br><br>
Normalization: <br><br>
MinMaxScaler was used to scale the features to a specific range (0, 1). Normalizing the data helps in stabilizing the training process and improving convergence.
<br><br>
Handling Missing Values: Missing values were filled with the mean of the column to ensure that the dataset is complete before training the models.
<br><br>
Feature Engineering: <br><br>No explicit feature engineering was performed in the provided code. However, extracting relevant features from the time-series data can help in capturing meaningful patterns and trends. Feature engineering techniques such as rolling window statistics, technical indicators (e.g., moving averages, RSI), and lagged features can be explored to capture the underlying patterns in the data better.

- - **Model Overfitting**:<br><br>
To prevent overfitting while training the models:
<br><br>
- - Train-Test Split:
<br><br>The dataset was split into training and testing sets to evaluate the model's performance on unseen data.
<br><br>
- - Regularization:<br><br> No explicit regularization techniques like dropout or L2 regularization were used in the provided code. Incorporating dropout layers or L2 regularization can help in preventing overfitting and improving the model's generalization capabilities.
<br><br>
- - Model Complexity:<br><br> The models used in the provided code (RNN, GRU LSTM, BiLSTM) have a relatively simple architecture. Experimenting with different architectures, adding more layers, or adjusting the number of units can help in capturing the underlying patterns in the data without overfitting.
<br><br>
- - Hyperparameter Tuning: <br><br>Hyperparameters like learning rate, batch size, and number of epochs were not explicitly optimized in the provided code. Hyperparameter tuning using techniques like grid search or random search can help in finding the optimal set of hyperparameters that yield the best performance without overfitting.



- **Sequence Generation**: The dataset is split into sequences of fixed length, with each sequence representing a window of historical data. This step is crucial for training the sequential models like RNN, LSTM, BiLSTM, and GRU.
- 
## Deep Learning Models
The following deep learning models are implemented for stock price prediction:
1. **Recurrent Neural Network (RNN)**: A basic RNN model is trained on the sequence data to capture temporal dependencies.
2. **Long Short-Term Memory (LSTM)**: LSTM networks are used to overcome the vanishing gradient problem in traditional RNNs and better capture long-term dependencies.
3. **Bidirectional LSTM (BiLSTM)**: BiLSTM networks are employed to leverage both past and future information for prediction.
4. **Gated Recurrent Unit (GRU)**: Similar to LSTM, GRU networks are designed to capture long-term dependencies but with fewer parameters.

## Training and Evaluation
The dataset is split into training and testing sets, with a portion of the data reserved for validation during training. Each model is trained using the training data and evaluated using the testing data. Evaluation metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are calculated to assess the performance of each model.

## Visualization
To visualize the performance of the models, the predicted stock prices are compared against the actual prices using line plots. Additionally, other visualization techniques such as candlestick charts may be employed to provide a more detailed analysis of the predictions.




## Conclusion
In conclusion, this project demonstrates the effectiveness of various deep learning models in predicting stock prices. By preprocessing the data and utilizing models such as RNN, LSTM, BiLSTM, and GRU, accurate predictions can be made, aiding investors in making informed decisions. The visualization techniques employed provide insights into the performance of the models and aid in interpreting the results.

For further details and implementation, please refer to the code repository on GitHub.

**GitHub Repository Link**: [link]

**Author**: [Your Name]

**Date**: [Date]

**License**: [License information]
