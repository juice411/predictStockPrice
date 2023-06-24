import gym
from gym import spaces
import pandas as pd
import numpy as np
import requests
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from sklearn.model_selection import train_test_split
from stable_baselines3.common.callbacks import BaseCallback


class IterationCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(IterationCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        total_iterations = self.num_timesteps  # 总迭代次数
        remaining_iterations = self.locals["total_timesteps"] - total_iterations  # 剩余迭代次数
        print(f"Remaining iterations: {remaining_iterations}")
        return True


# 读取股价数据和纳斯达克指数数据
# stock_data = pd.read_csv('microsoft_stock.csv')
# nasdaq_data = pd.read_csv('nasdaq_index.csv')
r = requests.get("http://api.finance.ifeng.com/akdaily/?code={}&type=last".format('sz002865'))
stock_data = pd.DataFrame(r.json()['record'],
                          columns=['date', 'open', 'high', 'close', 'low', 'volume', 'chg', 'pchg', 'ma5', 'ma10',
                                   'ma20', 'vma5', 'vma10', 'vma20', 'turnover'])
print(stock_data)

r = requests.get("http://api.finance.ifeng.com/akdaily/?code={}&type=last".format('sz399001'))
nasdaq_data = pd.DataFrame(r.json()['record'],
                           columns=['date', 'open_', 'high_', 'close_', 'low_', 'volume_', 'chg_', 'pchg_', 'ma5_',
                                    'ma10_',
                                    'ma20_', 'vma5_', 'vma10_', 'vma20_'])

# 合并股价数据和纳斯达克指数数据
data = pd.merge(stock_data, nasdaq_data, on='date')

# 将date_column列值从字符串转换为日期类型
data['date'] = pd.to_datetime(data['date'])


# 去除逗号并转换为数字类型的函数
def remove_comma_and_convert_to_numeric(value):
    if isinstance(value, str):
        return pd.to_numeric(value.replace(",", ""), errors='coerce')
    return value


# 对DataFrame的所有列（除了'Date'列）应用去逗号并转换为数字类型的操作
data = data.applymap(remove_comma_and_convert_to_numeric)

print(data)

# 添加特征参数
# data['Moving_Average_5'] = data['ma5']
# data['Moving_Average_10'] = data['ma10']
# data['Moving_Average_20'] = data['ma20']
# data['Price_Change'] = data['pchg']
# data['Volume'] = data['volume']


# 添加其他特征参数

# 提取特征和目标变量
features = data[
    ['open', 'high', 'close', 'low', 'volume', 'pchg', 'ma5', 'ma10', 'ma20', 'vma5', 'vma10', 'vma20', 'turnover',
     'open_', 'high_', 'close_', 'low_', 'volume_', 'pchg_', 'ma5_',
     'ma10_', 'ma20_', 'vma5_', 'vma10_', 'vma20_']].values
target = data['close'].values

# 划分训练集和测试集
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2,
                                                                            shuffle=False)


# 定义环境
class StockPredictionEnv(gym.Env):
    def __init__(self, features, target):
        super(StockPredictionEnv, self).__init__()
        self.features = features
        self.target = target
        self.current_step = 0
        self.max_steps = len(features)

        # 设置观察空间
        observation_low = -float('inf')
        observation_high = float('inf')
        self.observation_space = spaces.Box(observation_low, observation_high, shape=(25,), dtype=np.float32)
        # 设置动作空间为Box类型，指定每个动作维度的上限和下限
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.features[self.current_step]

    def step(self, action):
        if self.current_step == self.max_steps - 1:
            done = True
            reward = 0  # 最后一步时，可以选择不给予奖励
            next_state = np.zeros_like(self.features[0])  # 最后一步时，可以选择返回一个全零的状态
        else:
            done = False
            reward = self.target[self.current_step] - self.target[self.current_step - 1]  # 根据实际情况定义奖励函数
            self.current_step += 1
            next_state = self.features[self.current_step]

        return next_state, reward, done, {}


env = StockPredictionEnv(train_features, train_target)

# 定义PPO模型
model = PPO("MlpPolicy", env, verbose=1)

# 定义迭代次数回调函数
callback = IterationCallback()

# 训练模型
model.learn(total_timesteps=10000, callback=callback)

# 保存模型
model.save("stock_prediction_model")

# 加载模型
loaded_model = PPO.load("stock_prediction_model")

# 定义测试集环境
test_env = StockPredictionEnv(test_features, test_target)

# 在测试集上进行预测
obs = test_env.reset()
done = False
predicted_prices = []
while not done:
    action, _ = loaded_model.predict(obs)
    obs, reward, done, _ = env.step(action)
    predicted_prices.append(obs[2])  # 假设股价是特征向量的第一个元素

# 可视化回测结果
plt.plot(test_target, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction Backtest')
plt.legend()
plt.show()
