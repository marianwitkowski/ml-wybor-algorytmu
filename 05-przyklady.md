### 5. Zaawansowane przykłady praktyczne

W kontekście uczenia maszynowego istnieje wiele wyzwań związanych z wyborem odpowiednich algorytmów w zależności od charakterystyki problemu, danych oraz oczekiwanych wyników. W tej sekcji omówimy zaawansowane przykłady praktyczne, które obejmują wybór algorytmu dla problemów wieloklasowych (multi-class classification), stosowanie algorytmów regresji w przypadkach danych nieliniowych i nienormalnych rozkładów oraz analizę praktycznych aspektów algorytmów klasteryzacji dla dużych zbiorów danych.

#### 5.1. Wybór algorytmu dla problemu wieloklasowego (multi-class classification)

Wieloklasowa klasyfikacja polega na przypisywaniu obserwacji do jednej z wielu kategorii. W przeciwieństwie do klasyfikacji binarnej, gdzie mamy tylko dwie klasy, w klasyfikacji wieloklasowej istnieje więcej niż dwie klasy, co wprowadza dodatkową złożoność w wyborze i implementacji algorytmu.

##### Przykład: Klasyfikacja gatunków irysów

Rozważmy klasyczny problem klasyfikacji gatunków irysów na trzy różne kategorie: Iris Setosa, Iris Versicolor i Iris Virginica. Dane są stosunkowo małe, ale wieloklasowe, co czyni je idealnym przykładem do analizy wyboru odpowiedniego algorytmu.

**Algorytmy odpowiednie dla problemów wieloklasowych:**

- **Drzewa decyzyjne**: Dobrze radzą sobie z klasyfikacją wieloklasową, ponieważ naturalnie rozgałęziają się na podstawie cech i mogą obsługiwać wiele klas.
- **Support Vector Machines (SVM)**: Przy pomocy strategii *one-vs-one* lub *one-vs-all*, SVM może być używany do klasyfikacji wieloklasowej.
- **K-Nearest Neighbors (k-NN)**: Ten algorytm również może być używany do klasyfikacji wieloklasowej, szczególnie w przypadku mniejszych zbiorów danych.

**Przykładowy kod dla klasyfikacji wieloklasowej za pomocą drzew decyzyjnych:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Wczytywanie danych
iris = load_iris()
X = iris.data
y = iris.target

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu drzewa decyzyjnego
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

Wyniki klasyfikacji będą zawierały dokładność oraz szczegółowe metryki takie jak precyzja, czułość (recall) oraz F1-score dla każdej z klas.

**Przykładowy kod dla klasyfikacji wieloklasowej za pomocą SVM:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Wczytywanie danych
iris = load_iris()
X = iris.data
y = iris.target

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu SVM
model = SVC(kernel='linear', decision_function_shape='ovr')
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

SVM jest bardziej złożonym algorytmem, który może lepiej radzić sobie z bardziej skomplikowanymi granicami decyzyjnymi, ale może być również bardziej kosztowny obliczeniowo w porównaniu do prostszych drzew decyzyjnych.

#### 5.2. Algorytmy regresji w kontekście danych nieliniowych i nienormalnych rozkładów

Regresja to proces przewidywania wartości ciągłej na podstawie jednego lub więcej predyktorów (cech). W przypadku danych, które są nieliniowe lub mają nienormalny rozkład, klasyczne podejścia takie jak regresja liniowa mogą okazać się niewystarczające. W takich przypadkach stosuje się zaawansowane techniki regresji, które lepiej radzą sobie z tego typu danymi.

##### Przykład: Przewidywanie cen nieruchomości

Ceny nieruchomości mogą zależeć od wielu czynników, takich jak lokalizacja, wielkość, liczba pokoi, czy wiek budynku. Dane te często są nieliniowe i mogą mieć nienormalny rozkład, co sprawia, że tradycyjna regresja liniowa może być niewystarczająca.

**Algorytmy odpowiednie dla danych nieliniowych i nienormalnych:**

- **Regresja wielomianowa**: Rozszerzenie regresji liniowej, które pozwala na modelowanie nieliniowych relacji.
- **Las Losowy (Random Forest Regression)**: Algorytm zespołowy, który dobrze radzi sobie z nieliniowymi zależnościami.
- **Regresja metodą najbliższych sąsiadów (KNN Regression)**: Algorytm, który szuka najbliższych sąsiadów dla danego punktu danych i przewiduje wartość na podstawie wartości sąsiadów.

**Przykładowy kod dla regresji wielomianowej:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generowanie przykładowych danych
np.random.seed(0)
X = np.random.rand(100, 1) * 6 - 3  # Losowe dane w zakresie -3 do 3
y = X**3 - 3*X**2 + 2*X + np.random.randn(100, 1) * 3  # Funkcja nieliniowa + szum

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Transformacja wielomianowa
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Trening modelu regresji wielomianowej
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predykcje
y_pred = model.predict(X_test_poly)

# Ocena modelu
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Wizualizacja
plt.scatter(X_test, y_test, color='blue', label='Rzeczywiste')
plt.scatter(X_test, y_pred, color='red', label='Przewidywane')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresja Wielomianowa')
plt.legend()
plt.show()
```

W tym przypadku regresja wielomianowa pozwala na uchwycenie nieliniowych relacji w danych, co skutkuje bardziej dokładnymi przewidywaniami niż w przypadku regresji liniowej.

**Przykładowy kod dla regresji z użyciem Random Forest:**

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generowanie przykładowych danych
np.random.seed(0)
X = np.random.rand(100, 1) * 6 - 3
y = X**3 - 3*X**2 + 2*X + np.random.randn(100, 1) * 3

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.ravel())

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
mse = mean_squared_error(y_test, y_pred)
print

(f'Mean Squared Error: {mse}')
```

Random Forest potrafi efektywnie modelować nieliniowe relacje i jest mniej podatny na przeuczenie, co czyni go skutecznym narzędziem do przewidywania w przypadkach, gdy dane są nieliniowe i mają nienormalny rozkład.

#### 5.3. Praktyczne aspekty algorytmów klasteryzacji dla dużych zbiorów danych (np. k-means++, DBSCAN)

Klasteryzacja to technika grupowania danych w klastry, gdzie dane w tym samym klastrze są bardziej podobne do siebie niż do danych w innych klastrach. Jest szeroko stosowana w eksploracyjnej analizie danych, segmentacji rynku, wykrywaniu anomalii i innych zastosowaniach. Przy dużych zbiorach danych wybór odpowiedniego algorytmu klasteryzacji staje się kluczowy, ponieważ wiele z tych algorytmów może być wrażliwych na skalę danych.

##### Przykład: Segmentacja klientów na podstawie ich zachowań zakupowych

Załóżmy, że mamy duży zbiór danych opisujący zachowania zakupowe klientów. Naszym celem jest podzielenie klientów na segmenty, które będą miały podobne wzorce zakupowe, co może pomóc w tworzeniu spersonalizowanych kampanii marketingowych.

**Algorytmy odpowiednie dla klasteryzacji dużych zbiorów danych:**

- **k-means++**: Ulepszona wersja klasycznego algorytmu k-means, która lepiej radzi sobie z wyborem początkowych centroidów, co prowadzi do lepszego końcowego podziału.
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Algorytm oparty na gęstości, który potrafi wykrywać klastry o dowolnym kształcie i jest odporny na szum.

**Przykładowy kod dla k-means++:**

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generowanie przykładowych danych
X, _ = make_blobs(n_samples=1000, centers=5, cluster_std=0.6, random_state=42)

# Klasteryzacja z użyciem k-means++
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Wizualizacja klastrów
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroidy')
plt.title('Klasteryzacja k-means++')
plt.legend()
plt.show()
```

k-means++ minimalizuje problem z lokalnymi minimami, który jest częsty w klasycznym k-means, co jest szczególnie ważne przy pracy z dużymi zbiorami danych.

**Przykładowy kod dla DBSCAN:**

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generowanie przykładowych danych
X, _ = make_blobs(n_samples=1000, centers=3, cluster_std=0.5, random_state=42)

# Klasteryzacja z użyciem DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=10)
y_dbscan = dbscan.fit_predict(X)

# Wizualizacja klastrów
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis')
plt.title('Klasteryzacja DBSCAN')
plt.show()
```

DBSCAN jest bardzo skuteczny w wykrywaniu klastrów o różnych kształtach i jest odporny na obecność szumu, co sprawia, że jest idealny dla dużych, złożonych zbiorów danych.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone