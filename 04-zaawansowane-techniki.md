### 4. Zaawansowane techniki walidacji i testowania modeli

Walidacja i testowanie modeli uczenia maszynowego to kluczowe etapy, które pozwalają na ocenę jakości modeli oraz ich zdolności do generalizacji. W tym punkcie omówimy zaawansowane techniki walidacji i testowania, w tym walidację krzyżową z dostosowanymi strategiami, techniki resamplingu takie jak bootstrapping i jackknife, oraz dobór optymalnych metryk oceny modelu dla konkretnych problemów. Zastosowanie tych technik pozwala na uzyskanie bardziej wiarygodnych wyników oraz na lepsze zrozumienie, jak model będzie się zachowywać w rzeczywistych warunkach.

#### 4.1. Walidacja krzyżowa z dostosowanymi strategiami pod kątem specyfiki danych

Walidacja krzyżowa to jedna z najpopularniejszych technik oceny modelu, która pozwala na uzyskanie bardziej wiarygodnych wyników niż pojedynczy podział na zbiór treningowy i testowy. Polega ona na wielokrotnym dzieleniu danych na różne podzbiory treningowe i testowe oraz na trenowaniu modelu na różnych częściach danych. Następnie średnia z wyników uzyskanych na poszczególnych podzbiorach jest używana jako ocena ogólnej wydajności modelu.

##### K-Fold Cross-Validation

Najczęściej stosowaną techniką walidacji krzyżowej jest **K-Fold Cross-Validation**, w której dane są dzielone na K podzbiorów (ang. *folds*). Model jest trenowany na K-1 podzbiorach, a następnie testowany na pozostałym podzbiorze. Proces ten jest powtarzany K razy, a każda część danych pełni rolę zbioru testowego dokładnie raz.

**Przykładowy kod dla K-Fold Cross-Validation:**

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('dataset.csv')

# Przygotowanie danych
X = data.drop('target', axis=1)
y = data['target']

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# K-Fold Cross-Validation
scores = cross_val_score(model, X, y, cv=10)  # 10-fold cross-validation
print(f'Scores: {scores}')
print(f'Mean Accuracy: {scores.mean()}')
```

Wynikiem tej walidacji będzie zestaw 10 wyników dokładności (accuracy), które mogą być uśrednione, aby uzyskać bardziej wiarygodny wynik dotyczący wydajności modelu.

##### Stratified K-Fold Cross-Validation

**Stratyfikowana K-Fold Cross-Validation** jest odmianą K-Fold Cross-Validation, która zapewnia, że każda część danych (fold) ma taki sam rozkład klas jak oryginalny zbiór danych. Jest to szczególnie ważne w przypadku niezbalansowanych zbiorów danych, gdzie niektóre klasy mogą być niedostatecznie reprezentowane.

**Przykładowy kod dla Stratified K-Fold Cross-Validation:**

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('dataset.csv')

# Przygotowanie danych
X = data.drop('target', axis=1)
y = data['target']

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=10)
scores = cross_val_score(model, X, y, cv=skf)
print(f'Scores: {scores}')
print(f'Mean Accuracy: {scores.mean()}')
```

Stratyfikacja zapewnia, że każda klasa jest reprezentowana w każdym foldzie, co pozwala na uzyskanie bardziej rzetelnych wyników w przypadku niezbalansowanych danych.

##### Leave-One-Out Cross-Validation (LOOCV)

**Leave-One-Out Cross-Validation (LOOCV)** to ekstremalna wersja walidacji krzyżowej, gdzie każdy fold składa się tylko z jednej próbki, a pozostałe próbki są używane do trenowania modelu. Proces ten jest powtarzany tyle razy, ile jest próbek w zbiorze danych. LOOCV jest szczególnie użyteczne w przypadku małych zbiorów danych, ale może być bardzo kosztowne obliczeniowo dla dużych zbiorów.

**Przykładowy kod dla LOOCV:**

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('dataset.csv')

# Przygotowanie danych
X = data.drop('target', axis=1)
y = data['target']

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Leave-One-Out Cross-Validation
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
print(f'Mean Accuracy: {scores.mean()}')
```

LOOCV zapewnia bardzo dokładną ocenę modelu, ale ze względu na swoją wysoką złożoność obliczeniową, jest rzadziej stosowany w praktyce, szczególnie przy dużych zbiorach danych.

#### 4.2. Techniki resamplingu: Bootstrapping, Jackknife

Techniki resamplingu są używane do oszacowania dokładności modelu poprzez wielokrotne losowanie próbek z oryginalnego zbioru danych i oceny modelu na tych próbkach. Dwie najpopularniejsze techniki to **bootstrapping** i **jackknife**.

##### Bootstrapping

**Bootstrapping** polega na wielokrotnym losowaniu próbek (z zamianą) z oryginalnego zbioru danych, tworzeniu na ich podstawie nowych zbiorów i trenowaniu modelu na tych nowych zbiorach. Ponieważ próbki są losowane z zamianą, ten sam przykład może pojawić się więcej niż raz w jednym zbiorze danych. Bootstrapping jest szczególnie użyteczny, gdy mamy ograniczoną ilość danych, ponieważ pozwala na stworzenie wielu zbiorów danych z jednego oryginalnego zestawu.

**Przykładowy kod dla Bootstrapping:**

```python
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('dataset.csv')

# Przygotowanie danych
X = data.drop('target', axis=1)
y = data['target']

# Bootstrapping
n_iterations = 100
n_size = int(len(X) * 0.5)  # Używamy 50% danych w każdym bootstrapie
scores = list()

for i in range(n_iterations):
    # Resampling z zamianą
    X_resample, y_resample = resample(X, y, n_samples=n_size)
    
    # Trening modelu
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_resample, y_resample)
    
    # Predykcja na pełnym zbiorze testowym
    y_pred = model.predict(X)
    score = accuracy_score(y, y_pred)
    scores.append(score)

# Ocena
print(f'Bootstrapping Mean Accuracy: {sum(scores) / len(scores)}')
```

Bootstrapping pozwala na uzyskanie rozkładu wyników modelu, co może być użyteczne do oszacowania niepewności i wariancji wyników.

##### Jackknife

**Jackknife** to technika resamplingu podobna do Leave-One-Out Cross-Validation, ale zamiast trenować model tyle razy, ile jest próbek, jackknife usuwa jedną próbkę na raz i trenuje model na pozostałych danych. Jackknife jest mniej obciążający obliczeniowo niż LOOCV i często jest stosowany do estymacji błędów w modelach.

**Przykładowy kod dla Jackknife:**

```python
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Wczytywanie danych
data = pd.read_csv('dataset.csv')

# Przygotowanie danych
X = data.drop('target', axis=1)
y = data['target']

# Jackknife
n_samples = len(X)
scores = list()

for i in range(n_samples):
    # Usuwanie jednej próbki
    X_jackknife = np.delete(X.values, i, axis=0)
    y_jackknife = np.delete(y.values, i, axis=0)
    
    # Trening modelu
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_jackknife, y_jackknife)
    
    # Predykcja

 na usuniętej próbce
    y_pred = model.predict([X.values[i]])
    score = accuracy_score([y.values[i]], y_pred)
    scores.append(score)

# Ocena
print(f'Jackknife Mean Accuracy: {sum(scores) / len(scores)}')
```

Jackknife jest szczególnie przydatny do oceny stabilności modelu i do wykrywania wpływu poszczególnych próbek na wynik modelu.

#### 4.3. Dobór optymalnych metryk oceny modelu dla konkretnych problemów

Wybór odpowiednich metryk oceny modelu jest kluczowy dla poprawnej interpretacji wyników. W zależności od rodzaju problemu (klasyfikacja, regresja) i specyfiki danych, różne metryki mogą być bardziej lub mniej odpowiednie.

##### Klasyfikacja

W zadaniach klasyfikacyjnych najczęściej używane metryki to:

- **Dokładność (Accuracy)**: Procent poprawnych klasyfikacji w stosunku do wszystkich próbek.
  
- **Precyzja (Precision)**: Procent prawidłowych pozytywnych klasyfikacji w stosunku do wszystkich pozytywnych klasyfikacji.
  
- **Czułość (Recall)**: Procent prawidłowych pozytywnych klasyfikacji w stosunku do wszystkich rzeczywistych pozytywnych próbek.
  
- **F1-Score**: Harmoniczna średnia precyzji i czułości, która jest szczególnie przydatna przy niezbalansowanych danych.

**Przykładowy kod dla oceny modelu klasyfikacyjnego:**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('classification_dataset.csv')

# Przygotowanie danych
X = data.drop('target', axis=1)
y = data['target']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
```

Dobór odpowiedniej metryki zależy od specyfiki problemu. Na przykład, w zadaniach medycznych, gdzie błędne klasyfikacje mogą mieć poważne konsekwencje, czułość (recall) może być bardziej istotna niż precyzja.

##### Regresja

W zadaniach regresyjnych najczęściej używane metryki to:

- **Mean Absolute Error (MAE)**: Średnia wartość absolutnych różnic między wartościami przewidywanymi a rzeczywistymi.
  
- **Mean Squared Error (MSE)**: Średnia wartość kwadratów różnic między wartościami przewidywanymi a rzeczywistymi.
  
- **Root Mean Squared Error (RMSE)**: Pierwiastek kwadratowy z MSE, który daje bardziej intuicyjną miarę błędu.
  
- **R-squared (R²)**: Miara, która określa, jak dobrze model przewiduje wyniki w stosunku do średniej wartości danych.

**Przykładowy kod dla oceny modelu regresyjnego:**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('regression_dataset.csv')

# Przygotowanie danych
X = data.drop('target', axis=1)
y = data['target']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R²: {r2}')
```

W przypadku regresji, RMSE jest często preferowane, ponieważ karze większe błędy bardziej niż MAE. R² natomiast jest miarą ogólnej jakości dopasowania modelu i może być szczególnie użyteczne, gdy chcemy porównać różne modele.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone