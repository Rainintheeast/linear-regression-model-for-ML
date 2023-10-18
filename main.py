import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 生成一些随机数据
np.random.seed(0)
X = 2 * np.random.rand(3000, 1)
y = 3 + 4 * X + np.random.randn(3000, 1)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 创建线性回归模型
lin_reg = LinearRegression()

# 训练模型
lin_reg.fit(X_train, y_train)

# 在测试集上评估模型
y_pred = lin_reg.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 输出模型参数
print(f'Intercept: {lin_reg.intercept_}')
print(f'Coefficient: {lin_reg.coef_}')
