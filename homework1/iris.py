import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Загрузка данных
iris = load_iris()
X = iris.data


# Определение оптимального количества кластеров с помощью метода "локтя" и силуэта
def find_optimal_clusters(X):
    wcss = []
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    return wcss, silhouette_scores


wcss, silhouette_scores = find_optimal_clusters(X)

# Визуализация метода "локтя" и коэффициента силуэта
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, 11), wcss, marker='o')
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Коэффициент силуэта')
plt.xlabel('Количество кластеров')
plt.ylabel('Silhouette Score')

plt.show()

# Оптимальное количество кластеров (например, выбираем по максимуму силуэта)
optimal_k = np.argmax(silhouette_scores) + 2
print(f'Оптимальное количество кластеров: {optimal_k}')

# K-means алгоритм
k = optimal_k

# Инициализация центроидов
np.random.seed(0)
centroids = X[np.random.choice(range(len(X)), k, replace=False)]


# Функция для вычисления расстояния между точками
def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# Функция для присвоения точек к кластерам
def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)


# Функция для обновления центроидов
def update_centroids(X, clusters, k):
    centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return np.array(centroids)


# Основной цикл алгоритма K-means
max_iter = 100

# Создание списка для хранения изображений
images = []

for iteration in range(max_iter):
    # Присвоение точек к кластерам
    clusters = assign_clusters(X, centroids)

    # Визуализация текущего состояния
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    for i in range(k):
        cluster_points = X[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i % len(colors)], label=f'Кластер {i + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', label='Центроиды')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(f'K-means Clustering - Итерация {iteration + 1}')
    plt.legend()

    # Сохранение текущего изображения в списке
    plt.savefig(f'iteration_{iteration}.png')
    plt.close()

    # Добавление текущего изображения в список изображений для создания GIF
    images.append(imageio.imread(f'iteration_{iteration}.png'))

    # Пересчет центроидов
    new_centroids = update_centroids(X, clusters, k)

    # Проверка сходимости
    if np.all(centroids == new_centroids):
        print("Алгоритм сходится на шаге", iteration + 1)
        break

    centroids = new_centroids

# Создание GIF изображения
imageio.mimsave('kmeans_animation.gif', images, fps=2)
