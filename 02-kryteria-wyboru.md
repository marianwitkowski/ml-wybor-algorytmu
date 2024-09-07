### 2. Zaawansowane kryteria wybierania algorytmu

Wybór algorytmu uczenia maszynowego to kluczowy etap w procesie tworzenia modelu, który może mieć istotny wpływ na jego ostateczną wydajność i skuteczność. Wybierając algorytm, specjaliści od uczenia maszynowego muszą brać pod uwagę nie tylko dokładność modelu, ale również takie aspekty jak złożoność obliczeniowa, skalowalność, równowaga między dokładnością a interpretowalnością oraz sposób zarządzania trade-offami, takimi jak Bias-Variance czy czas trenowania versus jakość modelu. W tej sekcji omówimy te zaawansowane kryteria wyboru algorytmu i zilustrujemy je przykładami kodów.

#### 2.1. Złożoność obliczeniowa i skalowalność w dużych zbiorach danych

Złożoność obliczeniowa odnosi się do ilości zasobów, takich jak czas procesora i pamięć, które są potrzebne do przetwarzania algorytmu. Skalowalność natomiast dotyczy zdolności algorytmu do efektywnego działania, gdy rozmiar danych rośnie. W przypadku dużych zbiorów danych, złożoność obliczeniowa i skalowalność stają się kluczowymi czynnikami przy wyborze algorytmu.

**Przykład: Algorytm k-Nearest Neighbors (k-NN) vs. Drzewa Decyzyjne**

Algorytm k-Nearest Neighbors (k-NN) jest intuicyjny i łatwy do implementacji, ale jego złożoność obliczeniowa rośnie wykładniczo wraz z rozmiarem danych. Przy dużych zbiorach danych, k-NN może stać się niepraktyczny, ponieważ wymaga porównania każdego nowego punktu danych ze wszystkimi istniejącymi punktami w zbiorze. Z kolei drzewa decyzyjne są znacznie bardziej skalowalne, ponieważ struktura drzewa pozwala na szybkie odnalezienie odpowiedniego węzła, co znacząco redukuje liczbę operacji.

**Przykładowy kod dla k-NN:**

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('large_dataset.csv')

# Przygotowanie danych
X = data.drop('label', axis=1)
y = data['label']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening modelu
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predykcje
y_pred = knn.predict(X_test)

# Ocena modelu
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
```

**Przykładowy kod dla drzewa decyzyjnego:**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('large_dataset.csv')

# Przygotowanie danych
X = data.drop('label', axis=1)
y = data['label']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening modelu
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Predykcje
y_pred = tree.predict(X_test)

# Ocena modelu
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
```

Wyniki pokazują, że choć oba algorytmy mogą osiągnąć porównywalną dokładność, drzewo decyzyjne jest znacznie bardziej skalowalne i efektywne przy dużych zbiorach danych.

#### 2.2. Równowaga między dokładnością a interpretowalnością

Interpretowalność modelu odnosi się do łatwości, z jaką możemy zrozumieć, jak model dochodzi do swoich wyników. Modele o wysokiej interpretowalności, takie jak regresja liniowa czy drzewa decyzyjne, są często preferowane w sytuacjach, gdzie konieczne jest zrozumienie logiki stojącej za decyzjami modelu. Z drugiej strony, bardziej złożone modele, takie jak gradient boosting czy Support Vector Machines, mogą oferować wyższą dokładność, ale kosztem interpretowalności.

**Przykład: Regresja Liniowa vs. Random Forest**

Regresja liniowa jest jednym z najbardziej interpretowalnych modeli, ponieważ wyraźnie pokazuje wpływ każdej cechy na wynik. Jednak w przypadku bardziej złożonych zależności między cechami, jej dokładność może być ograniczona. Random Forest, będący metodą zespołową opartą na wielu drzewach decyzyjnych, oferuje znacznie większą dokładność w takich przypadkach, ale trudniej jest zrozumieć dokładny wpływ poszczególnych cech na wynik.

**Przykładowy kod dla regresji liniowej:**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('house_prices.csv')

# Przygotowanie danych
X = data.drop('price', axis=1)
y = data['price']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening modelu
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predykcje
y_pred = lr.predict(X_test)

# Ocena modelu
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Współczynniki regresji
coefficients = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
```

Współczynniki regresji liniowej pozwalają na łatwą interpretację wpływu poszczególnych cech na cenę domu. Każdy współczynnik wskazuje, o ile zmieni się przewidywana cena domu, jeśli dana cecha wzrośnie o jednostkę, przy założeniu, że wszystkie inne cechy pozostają bez zmian.

**Przykładowy kod dla Random Forest:**

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('house_prices.csv')

# Przygotowanie danych
X = data.drop('price', axis=1)
y = data['price']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening modelu
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predykcje
y_pred = rf.predict(X_test)

# Ocena modelu
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Ważność cech
importances = pd.DataFrame(rf.feature_importances_, X.columns, columns=['Importance'])
print(importances.sort_values(by='Importance', ascending=False))
```

Random Forest oferuje możliwość oceny ważności cech, co daje pewien wgląd w to, które cechy są najważniejsze dla modelu. Jednak w porównaniu do regresji liniowej, interpretacja wpływu poszczególnych cech jest znacznie bardziej skomplikowana.

#### 2.3. Zarządzanie trade-offami: Bias-Variance, Czas trenowania vs. Jakość modelu

W uczeniu maszynowym jednym z najważniejszych trade-offów, którymi należy zarządzać, jest kompromis między błędem systematycznym (bias) a błędem losowym (variance). Bias odnosi się do błędu spowodowanego uproszczeniami w modelu, podczas gdy variance wynika z nadmiernej złożoności modelu, która prowadzi do przeuczenia (overfitting). Równoważenie tych dwóch aspektów jest kluczowe dla stworzenia dobrze działającego modelu.

**Przykład: Regresja liniowa (niski variance, wysoki bias) vs. Lasy losowe (niski bias, wysoki variance)**

Regresja liniowa zazwyczaj charakteryzuje się wysokim biasem, co oznacza, że może nie uchwycić skomplikowanych zależności w danych, ale ma niski variance, co sprawia, że jest mniej podatna na prze

uczenie. Z kolei lasy losowe mogą lepiej dopasować się do danych treningowych (niski bias), ale istnieje ryzyko, że będą miały wysoki variance, co może prowadzić do przeuczenia.

**Przykładowy kod dla regresji liniowej:**

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('house_prices.csv')

# Przygotowanie danych
X = data.drop('price', axis=1)
y = data['price']

# Trening modelu z walidacją krzyżową
lr = LinearRegression()
scores = cross_val_score(lr, X, y, cv=5, scoring='neg_mean_squared_error')
print(f'Mean Cross-Validation MSE: {-scores.mean()}')
```

Regresja liniowa może nie oferować najniższego błędu, ale jest bardziej odporna na zmienność wyników przy różnych zbiorach danych.

**Przykładowy kod dla lasu losowego:**

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('house_prices.csv')

# Przygotowanie danych
X = data.drop('price', axis=1)
y = data['price']

# Trening modelu z walidacją krzyżową
rf = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
print(f'Mean Cross-Validation MSE: {-scores.mean()}')
```

Lasy losowe zazwyczaj oferują niższy błąd, ale są bardziej podatne na różnice między zbiorami danych.

**Czas trenowania vs. Jakość modelu**

Kolejnym ważnym kompromisem, z którym spotykają się specjaliści ds. uczenia maszynowego, jest równoważenie czasu trenowania modelu z jego jakością. W niektórych przypadkach, jak np. w zastosowaniach w czasie rzeczywistym, czas trenowania może być krytyczny i preferowane są szybsze modele, nawet jeśli ich jakość (dokładność) jest nieco niższa.

**Przykład: SVM vs. Drzewa decyzyjne**

Support Vector Machines (SVM) mogą oferować bardzo wysoką dokładność, zwłaszcza w przypadkach, gdy istnieją nieliniowe zależności w danych. Jednak czas trenowania SVM jest często dłuższy, zwłaszcza przy dużych zbiorach danych. Drzewa decyzyjne są znacznie szybsze do trenowania, choć mogą nie osiągnąć tej samej dokładności co SVM.

**Przykładowy kod dla SVM:**

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import time

# Wczytywanie danych
data = pd.read_csv('large_dataset.csv')

# Przygotowanie danych
X = data.drop('label', axis=1)
y = data['label']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening modelu
start_time = time.time()
svc = SVC()
svc.fit(X_train, y_train)
end_time = time.time()

# Predykcje
y_pred = svc.predict(X_test)

# Ocena modelu
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Czas trenowania: {end_time - start_time} sekund')
```

**Przykładowy kod dla drzewa decyzyjnego:**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import time

# Wczytywanie danych
data = pd.read_csv('large_dataset.csv')

# Przygotowanie danych
X = data.drop('label', axis=1)
y = data['label']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening modelu
start_time = time.time()
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
end_time = time.time()

# Predykcje
y_pred = tree.predict(X_test)

# Ocena modelu
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Czas trenowania: {end_time - start_time} sekund')
```

Jak widać, drzewa decyzyjne są znacznie szybsze w trenowaniu, co może być kluczowe w sytuacjach, gdzie czas jest istotnym czynnikiem.

#### 2.4. Algorytmy zoptymalizowane pod kątem przetwarzania na sprzęcie GPU/TPU

Złożone modele uczenia maszynowego, zwłaszcza te stosowane w dużych zbiorach danych, mogą wymagać znacznych zasobów obliczeniowych. Wykorzystanie jednostek GPU (Graphics Processing Unit) lub TPU (Tensor Processing Unit) pozwala na znaczne przyspieszenie obliczeń, co jest szczególnie istotne przy trenowaniu algorytmów na dużych zbiorach danych.

**Przykład: XGBoost**

XGBoost to jeden z algorytmów, który został zoptymalizowany do pracy na GPU. Dzięki temu może on trenować modele znacznie szybciej w porównaniu do tradycyjnych implementacji opartych na CPU.

**Przykładowy kod dla XGBoost z wykorzystaniem GPU:**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('large_dataset.csv')

# Przygotowanie danych
X = data.drop('label', axis=1)
y = data['label']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening modelu XGBoost z GPU
params = {
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist'
}
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

bst = xgb.train(params, dtrain, num_boost_round=100)

# Predykcje
y_pred = bst.predict(dtest)

# Ocena modelu
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

Dzięki wykorzystaniu GPU, XGBoost może trenować modele znacznie szybciej, co jest istotne przy pracy z dużymi zbiorami danych lub przy złożonych modelach.

**Przykład: CatBoost**

CatBoost to kolejny algorytm, który został zoptymalizowany pod kątem wykorzystania GPU. Jest szczególnie efektywny w przypadku danych kategorycznych.

**Przykładowy kod dla CatBoost z wykorzystaniem GPU:**

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('large_dataset.csv')

# Przygotowanie danych
X = data.drop('label', axis=1)
y = data['label']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening modelu CatBoost z GPU
model = CatBoostClassifier(iterations=1000, depth=10, learning_rate=0.1, task_type="GPU", devices='0')
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
```

Wykorzystanie GPU w CatBoost pozwala na znacznie szybsze trenowanie modeli, co jest szczególnie przydatne przy pracy z dużymi i złożonymi zbiorami danych.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone