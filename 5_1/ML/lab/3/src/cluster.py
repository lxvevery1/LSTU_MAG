from sklearn.metrics import confusion_matrix, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def compare(y_true, y_pred, target_names=None):
    # Ensure y_true and y_pred are numeric
    if y_true.dtype == object:
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
        if target_names is None:
            target_names = le.classes_

    cm = confusion_matrix(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    print("Confusion Matrix (True labels vs Clusters):")
    print(cm)
    print(f"\nAdjusted Rand Index (ARI): {ari:.3f}")

    return cm, ari

def plt_kmeans():
    plt.figure(figsize=(6,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_kmeans, cmap='viridis')
    plt.title(f"K-Means clustering (k={cluster_count})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

def plt_ward_tree():
    plt.figure(figsize=(10, 6))
    dendrogram(Z)
    plt.title("Hierarchical clustering (Ward's method)")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()

def plt_ward_clust():
    plt.figure(figsize=(6,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_ward, cmap='plasma')
    plt.title(f"Hierarchical clustering (Ward, { cluster_count } clusters)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

def arg_influence():
    for k in [2, 3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        plt.scatter(X_pca[:,0], X_pca[:,1], c=km.labels_, cmap='viridis')
        plt.title(f"KMeans (k={k})")
        plt.show()

def elbow():
    inertias = []
    k_values = range(1, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(6,4))
    plt.plot(k_values, inertias, marker='o')
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()




# 1. Постройте кластеризацию данных для определния вида ирисов
# (база Iris) на основе четырех параметров (без использования информации
# о классе). Для кластеризации используйте метод k-средних и иерархический
# метод Уорда. Исследуйте влияние аргументов процедур, реализующих указанные
# выше методы, на результат кластеризации. Постройте графики.

import pandas as pd

iris = pd.read_csv('iris.csv')
print(iris)

from sklearn.preprocessing import StandardScaler

y_true = iris['target']

# unsupervised learning
X = iris.drop(columns=['target'], errors='ignore')

# 1. K-mean
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cluster_count = 3
kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=5)
kmeans.fit(X_scaled)
labels_kmeans = kmeans.labels_

compare(y_true, labels_kmeans, target_names=iris['target'].unique())

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt_kmeans()

# 2. Ward
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

Z = linkage(X_scaled, method='ward')

plt_ward_tree()

criterion1 = 'maxclust'
criterion2 = 'inconsistent'
criterion3 = 'distance'

depth = 5
# criterion4 = 'monocrit'
# criterion5 = 'maxclust_monocrit'
labels_ward = fcluster(Z, cluster_count, criterion=criterion1, depth=depth)

compare(y_true, labels_ward, target_names=iris['target'].unique())

# 3. Args
arg_influence()

# 2. Опрееделите оптимальное количество кластеров
# на основе метода "локтя".
elbow()

# 3. Сравните результаты кластеризации для случая 3-х кластеров
# с реальными классами, представленными в базе данных Iris

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

X = iris.drop(columns=['target'], errors='ignore')
y_true = iris['target'] if 'target' in iris.columns else None

# Нормализация данных (для KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
y_kmeans = kmeans.labels_

le = LabelEncoder()
y_true_encoded = le.fit_transform(y_true)

compare(y_true_encoded, y_kmeans)

print("\nКоды классов:")
for i, cls in enumerate(le.classes_):
    print(f"{i}: {cls}")
