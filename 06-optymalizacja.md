### 6. Optymalizacja i tuning algorytmów

Optymalizacja i tuning algorytmów to kluczowe etapy w procesie uczenia maszynowego. Nawet najlepszy algorytm nie zapewni optymalnych wyników, jeśli nie zostanie odpowiednio dostrojony. Hiperparametry, czyli parametry sterujące działaniem algorytmu, muszą być precyzyjnie dostosowane, aby model osiągnął najwyższą możliwą wydajność. W tej sekcji omówimy zaawansowane techniki optymalizacji i tuningu hiperparametrów, takie jak Grid Search, Random Search oraz Bayesian Optimization. Przedstawimy również znaczenie regularyzacji i normalizacji w kontekście poprawy wydajności modeli oraz zaprezentujemy techniki automatyzacji procesu tuningu.

#### 6.1. Zaawansowane techniki Grid Search, Random Search, i Bayesian Optimization

##### Grid Search

**Grid Search** to jedna z najprostszych i najpopularniejszych technik optymalizacji hiperparametrów. Polega na przeszukiwaniu siatki zdefiniowanych wartości dla różnych hiperparametrów i trenowaniu modelu dla każdej możliwej kombinacji. Jest to metoda wszechstronna, ale jej wadą jest wysokie zapotrzebowanie na zasoby obliczeniowe, zwłaszcza gdy liczba hiperparametrów jest duża.

**Przykładowy kod dla Grid Search:**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Definiowanie modelu
model = RandomForestClassifier()

# Definiowanie siatki hiperparametrów
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Wyniki
print(f'Najlepsze parametry: {grid_search.best_params_}')
print(f'Najlepszy wynik: {grid_search.best_score_}')
```

W powyższym przykładzie używamy Grid Search do znalezienia najlepszych hiperparametrów dla klasyfikatora Random Forest. Siatka przeszukuje różne wartości dla `n_estimators`, `max_depth` i `min_samples_split`, aby wybrać najlepsze parametry.

##### Random Search

**Random Search** to technika optymalizacji hiperparametrów, która zamiast przeszukiwać wszystkie możliwe kombinacje, wybiera losowo podzbiór kombinacji. Dzięki temu, w porównaniu do Grid Search, Random Search może być bardziej efektywny pod względem obliczeniowym, a jednocześnie często daje porównywalne wyniki.

**Przykładowy kod dla Random Search:**

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Definiowanie modelu
model = RandomForestClassifier()

# Definiowanie zakresu hiperparametrów
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
    'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Random Search
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Wyniki
print(f'Najlepsze parametry: {random_search.best_params_}')
print(f'Najlepszy wynik: {random_search.best_score_}')
```

Random Search pozwala na losowe próbkowanie hiperparametrów z zdefiniowanego zakresu. Choć nie przeszukuje wszystkich kombinacji, to dzięki losowemu wyborowi może znaleźć dobre rozwiązania przy mniejszym koszcie obliczeniowym.

##### Bayesian Optimization

**Bayesian Optimization** to bardziej zaawansowana metoda optymalizacji, która buduje probabilistyczny model funkcji celu (najczęściej model Gaussian Process), a następnie wybiera kolejne zestawy hiperparametrów na podstawie poprzednich wyników. Bayesian Optimization jest znacznie bardziej efektywny niż Grid Search i Random Search, ponieważ inteligentnie wybiera kolejne próby, co prowadzi do szybszego znalezienia optymalnych hiperparametrów.

**Przykładowy kod dla Bayesian Optimization (z wykorzystaniem biblioteki `scikit-optimize`):**

```python
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Definiowanie modelu
model = RandomForestClassifier()

# Definiowanie zakresu hiperparametrów dla Bayesian Optimization
param_space = {
    'n_estimators': (100, 1000),
    'max_depth': (10, 110),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 4),
    'bootstrap': [True, False]
}

# Bayesian Optimization
opt = BayesSearchCV(model, param_space, n_iter=32, cv=5)
opt.fit(X_train, y_train)

# Wyniki
print(f'Najlepsze parametry: {opt.best_params_}')
print(f'Najlepszy wynik: {opt.best_score_}')
```

Bayesian Optimization iteracyjnie buduje model prawdopodobieństwa funkcji celu na podstawie wyników z poprzednich iteracji, co pozwala na bardziej efektywne poszukiwanie optymalnych hiperparametrów.

#### 6.2. Znaczenie regularyzacji i normalizacji: L1, L2, Elastic Net

W uczeniu maszynowym, regularyzacja jest techniką stosowaną w celu zapobiegania przeuczeniu (overfitting) poprzez dodanie kary za duże wartości współczynników modelu. Dwie najczęściej stosowane formy regularyzacji to **L1** i **L2**. Normalizacja danych, z drugiej strony, polega na przekształceniu cech do tej samej skali, co jest szczególnie ważne w przypadku algorytmów, które są wrażliwe na różne zakresy cech, takich jak regresja liniowa, SVM czy k-NN.

##### L1 Regularyzacja (Lasso)

**L1 Regularyzacja**, znana również jako *Lasso Regression*, dodaje karę za sumę wartości bezwzględnych współczynników. To prowadzi do sparsowania modelu, ponieważ wiele współczynników jest zerowanych, co oznacza, że pewne cechy są eliminowane z modelu.

**Przykładowy kod dla Lasso Regression:**

```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generowanie przykładowych danych
np.random.seed(0)
X = np.random.rand(100, 10)
y = X[:, 0] * 5 + X[:, 1] * -2 + np.random.randn(100)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Predykcje
y_pred = lasso.predict(X_test)

# Ocena modelu
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Współczynniki regresji: {lasso.coef_}')
```

Wynikiem Lasso będzie model z wieloma zerowymi współczynnikami, co oznacza, że niektóre cechy zostały wyeliminowane z modelu.

##### L2 Regularyzacja (Ridge)

**L2

 Regularyzacja**, znana również jako *Ridge Regression*, dodaje karę za sumę kwadratów współczynników. W przeciwieństwie do Lasso, Ridge nie zeruje współczynników, ale zmniejsza ich wartości, co zmniejsza wpływ mało istotnych cech na wynik modelu.

**Przykładowy kod dla Ridge Regression:**

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generowanie przykładowych danych
np.random.seed(0)
X = np.random.rand(100, 10)
y = X[:, 0] * 5 + X[:, 1] * -2 + np.random.randn(100)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Predykcje
y_pred = ridge.predict(X_test)

# Ocena modelu
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Współczynniki regresji: {ridge.coef_}')
```

Ridge Regression prowadzi do bardziej stabilnych współczynników niż Lasso, co jest szczególnie przydatne, gdy wszystkie cechy mają pewną istotność.

##### Elastic Net

**Elastic Net** łączy cechy zarówno L1, jak i L2 regularyzacji. Jest to model, który może wyeliminować pewne cechy (jak Lasso), ale jednocześnie utrzymać stabilność współczynników (jak Ridge).

**Przykładowy kod dla Elastic Net:**

```python
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generowanie przykładowych danych
np.random.seed(0)
X = np.random.rand(100, 10)
y = X[:, 0] * 5 + X[:, 1] * -2 + np.random.randn(100)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu Elastic Net
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)

# Predykcje
y_pred = elastic_net.predict(X_test)

# Ocena modelu
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Współczynniki regresji: {elastic_net.coef_}')
```

Elastic Net to elastyczne podejście, które może łączyć zalety obu metod regularyzacji, w zależności od potrzeb.

#### 6.3. Techniki automatyzacji procesu tuningu hiperparametrów

Ręczny tuning hiperparametrów może być czasochłonny i kosztowny obliczeniowo. Dlatego rozwinięto techniki automatyzacji tego procesu. Jedną z najbardziej popularnych technik jest **AutoML** (Automated Machine Learning), który automatyzuje cały proces od przygotowania danych po optymalizację hiperparametrów.

##### Przykład: Auto-sklearn

`auto-sklearn` to narzędzie w Pythonie oparte na `scikit-learn`, które automatycznie wybiera najlepszy model i optymalizuje hiperparametry za pomocą zaawansowanych technik optymalizacji, takich jak Bayesian Optimization.

**Przykładowy kod dla AutoML z użyciem `auto-sklearn`:**

```python
import autosklearn.classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# AutoML - automatyczny wybór modelu i hiperparametrów
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30)
automl.fit(X_train, y_train)

# Predykcje
y_pred = automl.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

`auto-sklearn` automatyzuje cały proces budowy modelu i optymalizacji hiperparametrów, co pozwala na oszczędność czasu i zasobów.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone
