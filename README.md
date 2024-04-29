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

![6](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/db150095-5223-4fd1-a4b9-63d777ef7048)


## Deep Learning Models
The following deep learning models are implemented for stock price prediction:
1. **Recurrent Neural Network (RNN)**: A basic RNN model is trained on the sequence data to capture temporal dependencies.
   ![7](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/166728c6-f2bb-4192-bc7f-a672b74fcb57)

2. **Long Short-Term Memory (LSTM)**: LSTM networks are used to overcome the vanishing gradient problem in traditional RNNs and better capture long-term dependencies.
   ![8](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/03a61651-ef48-4d45-96a3-5af921b32a32)

3. **Bidirectional LSTM (BiLSTM)**: BiLSTM networks are employed to leverage both past and future information for prediction.
   ![9](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/f676ef98-4fd5-42a6-b719-d0ed065435b3)

4. **Gated Recurrent Unit (GRU)**: Similar to LSTM, GRU networks are designed to capture long-term dependencies but with fewer parameters.
   ![10](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/92a3fa7d-ce5a-4682-affa-18c99c44a87a)


## Training and Testing
The dataset is split into training and testing sets, with a portion of the data reserved for validation during training. Each model is trained using the training data and evaluated using the testing data. Evaluation metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (MSE) are calculated to assess the performance of each model.

![11](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/f44943c6-7d72-44ad-b997-8d0abbc546e0)

## Performance Measures

The performance of the RNN model can be assessed based on the following regression metrics:
<br><br>

- - Mean Squared Error (MSE): <br><br>It measures the average squared difference between the predicted and actual values. Lower values of MSE indicate better model performance.
<br><br>
- - Mean Absolute Error (MAE): <br><br>It measures the average absolute difference between the predicted and actual values. Lower values of MAE indicate better model performance.
<br><br>
- - R-squared (R^2):<br><br> It represents the proportion of the variance for the dependent variable that's explained by independent variables in the model. A higher value of R^2 indicates a better fit of the model to the data.
 
## Models Evaluation

- GRU
- - GRU Model Loss: 0.00011<br>
- - MSE: 78530.71<br>
- - MAE: 201.89<br>
- - R^2: 99.9<br>

![image](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/44cccd16-bbf8-4346-b942-57c035c7f2f2)

- BiLSTM
- - BiLSTM Model Loss: 0.00049<br>
- - MSE: 323597.97<br>
- - MAE: 405.98<br>
- - R^2: 99.57<br>

![image](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/dc228fca-2305-4424-9c05-41c90fb99b49)

- RNN
- - RNN Model Loss: 0.0058
- - MSE: 3833435.63
- - MAE: 1063.58
- - R^2: 94.93

![image](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/92e035be-d531-41b7-860c-805021f3a13d)


- LSTM
- - LSTM Model Loss: 0.03<br>
- - MSE: 19679653.08<br>
- - MAE: 2538.23<br>
- - R^2: 73.99<br>

![image](https://github.com/s-mujtaba-haider/AI-Project/assets/110555927/0a9944b8-324b-424f-8aae-a553f32d3046)


**All** **the** **models** **are** **train** **on** **batch_size** **=** **64** **and** **epoch** **=** **300**
## Visualization
To visualize the performance of the models, the predicted stock prices are compared against the actual prices using line plots. Additionally, other visualization techniques such as candlestick charts may be employed to provide a more detailed analysis of the predictions.




## Conclusion
In conclusion, this project demonstrates the effectiveness of various deep learning models in predicting stock prices. By preprocessing the data and utilizing models such as RNN, LSTM, BiLSTM, and GRU, accurate predictions can be made, aiding investors in making informed decisions. The visualization techniques employed provide insights into the performance of the models and aid in interpreting the results.

For further details and implementation, please refer to the code repository on GitHub.

**GitHub Repository Link**: [link]

**Author**: [Your Name]

**Date**: [Date]

**License**: [License information]
