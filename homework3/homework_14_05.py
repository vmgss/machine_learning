import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def simple_linear_regression(X, y):
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Оценка качества модели на тестовом наборе
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Среднеквадратическая ошибка :", mse)

    # Построение прямой линии регрессии
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Погодные условия')
    plt.ylabel('Количество арендованных велосипедов')
    plt.title('Линейная регрессия')
    plt.show()


def predict_weather_dependence(X, y):
    # Создание и обучение модели
    model = LinearRegression()
    model.fit(X, y)

    # Предсказание количества арендованных велосипедов для всех значений оценки погоды
    weather_situations = np.array([[1], [2], [3], [4]])
    predicted_cnt_all = model.predict(weather_situations)
    for i, weather_situation in enumerate(weather_situations):
        print(f"Прогноз для аренды велосипедов в зависимости от погоды {weather_situation[0]}: {predicted_cnt_all[i]}")


def reduce_dimensionality(X, y):
    # Упрощаем модель с помощью PCA для улучшения качества предсказания количества арендованных велосипедов
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    x_pca = pca.fit_transform(X_scaled)

    # Преобразуем данные
    plt.scatter(x_pca, y, c=y, cmap='viridis')
    plt.colorbar(label='Количество арендованных велосипедов')  # Цветовая метка
    plt.xlabel('Погодные условия')
    plt.ylabel('Количество арендованных велосипедов')
    plt.title('Представление данных')
    plt.show()


def lasso_feature_influence(X, y):
    # Упрощаем модель с помощью Lasso
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_scaled, y)

    # Выводим коэффициенты признаков
    feature_coef = dict(zip(X.columns, lasso.coef_))
    print("Характерные коэффиценты:", feature_coef)

    # Находим признак с наибольшим влиянием на cnt
    most_influential_feature = max(feature_coef, key=feature_coef.get)
    print("Наиболее влиятельный признак:", most_influential_feature)


# Загрузка данных
data = pd.read_csv('../bikes_rent.csv')

# Выбор признаков
X = data[['weathersit']]
y = data['cnt']

# 2 пункт
simple_linear_regression(X, y)

# 3 и 4 пункты
predict_weather_dependence(X, y)
reduce_dimensionality(X, y)

# 5 пункт
lasso_feature_influence(X, y)
