import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

# 读取股价数据和纳斯达克指数数据
stock_data = pd.read_csv('microsoft_stock.csv')
nasdaq_data = pd.read_csv('nasdaq_index.csv')

# 合并股价数据和纳斯达克指数数据
data = pd.merge(stock_data, nasdaq_data, on='Date')

# 添加特征参数
data['Moving_Average_5'] = data['Close'].rolling(window=5).mean()
data['Moving_Average_10'] = data['Close'].rolling(window=10).mean()
data['Price_Change'] = data['Close'].pct_change()
data['Volume'] = data['Volume']
# 添加其他特征参数

# 提取特征和目标变量
features = data[['Moving_Average_5', 'Moving_Average_10', 'Price_Change', 'Volume', 'Nasdaq_Index']].values
target = data['Close'].values

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target.reshape(-1, 1))

# 划分训练集和测试集
train_size = int(len(data) * 0.8)
train_features = scaled_features[:train_size, :]
train_target = scaled_target[:train_size, :]
test_features = scaled_features[train_size:, :]
test_target = scaled_target[train_size:, :]

# 构建神经网络模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_features.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(train_features.reshape(train_features.shape[0], train_features.shape[1], 1), train_target, epochs=20, batch_size=32)

# 在测试集上进行预测
predictions = model.predict(test_features.reshape(test_features.shape[0], test_features.shape[1], 1))
predictions = scaler.inverse_transform(predictions)

# 打印预测结果
for i in range(len(predictions)):
    print('预测价格：', predictions[i][0], ' 实际价格：', scaler.inverse_transform(test_target)[i][0])
