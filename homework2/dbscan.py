import pygame


# Вычисление евклидова расстояния между двумя точками
def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


# Функция для поиска соседей точки в пределах eps
def find_neighbors(points, current_point, epsilon):
    neighbors = []
    for neighbor in points:
        if calculate_distance(current_point, neighbor) < epsilon:
            neighbors.append(neighbor)
    return neighbors


# Функция для расширения кластера
def grow_cluster(points, current_point, neighbors, cluster_id, epsilon, min_points, cluster_labels):
    cluster_labels[points.index(current_point)] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor = neighbors[i]
        if cluster_labels[points.index(neighbor)] == -1:
            cluster_labels[points.index(neighbor)] = cluster_id
        elif cluster_labels[points.index(neighbor)] == 0:
            cluster_labels[points.index(neighbor)] = cluster_id
            new_neighbors = find_neighbors(points, neighbor, epsilon)
            if len(new_neighbors) >= min_points:
                neighbors += new_neighbors
        i += 1


# Алгоритм DBSCAN
def perform_dbscan(points, epsilon, min_points):
    cluster_id = 0
    cluster_labels = [0] * len(points)
    for current_point in points:
        if cluster_labels[points.index(current_point)] == 0:
            neighbors = find_neighbors(points, current_point, epsilon)
            if len(neighbors) < min_points:
                cluster_labels[points.index(current_point)] = -1
            else:
                cluster_id += 1
                grow_cluster(points, current_point, neighbors, cluster_id, epsilon, min_points, cluster_labels)
    return cluster_labels


# Функция для добавления цвета
def get_color(label, base_color):
    if label == -1:
        return RED
    else:
        return (base_color[0] // 2, base_color[1] // 2, base_color[2] // 2)


# Константы для Pygame
WIDTH = 360
HEIGHT = 480
FPS = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
colors = [RED, GREEN, BLUE, YELLOW]

# Инициализация Pygame
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DBSCAN Clustering")
clock = pygame.time.Clock()
points = []
epsilon, min_points = 30, 2

# Основной цикл
running = True
while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                point = event.pos
                points.append(point)
                pygame.draw.circle(screen, WHITE, point, 5)

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                cluster_labels = perform_dbscan(points, epsilon, min_points)
                print(cluster_labels, points)
                # Отрисовка точек на экране с учетом кластеров
                for point, label in zip(points, cluster_labels):
                    color = get_color(label, WHITE)
                    pygame.draw.circle(screen, color, point, 5)
            if event.key == pygame.K_r:
                # Отрисовка точек с флажками (красный для шума, зеленый и желтый для граничных и ядерных точек)
                for point, label in zip(points, cluster_labels):
                    if label == -1:
                        pygame.draw.circle(screen, RED, point, 5)
                    elif label == 0:
                        pygame.draw.circle(screen, YELLOW, point, 5)
                    else:
                        pygame.draw.circle(screen, GREEN, point, 5)

    pygame.display.flip()
pygame.quit()
