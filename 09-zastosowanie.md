### 9. Przypadki zastosowań

Uczenie maszynowe znajduje zastosowanie w wielu branżach, gdzie różne algorytmy dostosowywane są do specyficznych wymagań i charakterystyki danych. W tym rozdziale omówimy kilka kluczowych przypadków zastosowań: finansowe modele ryzyka, diagnostyka medyczna oraz predykcje w marketingu. Skupimy się także na algorytmach specyficznych dla danych sekwencyjnych i przestrzennych, które są szczególnie użyteczne w określonych branżach. Przedstawimy także przykłady kodów implementujących odpowiednie algorytmy.

#### 9.1. Analiza przypadków: Finansowe modele ryzyka

Branża finansowa to jedno z kluczowych zastosowań dla uczenia maszynowego, gdzie algorytmy są wykorzystywane do modelowania ryzyka kredytowego, wykrywania oszustw i optymalizacji inwestycji. Modele predykcyjne pomagają instytucjom finansowym lepiej oceniać ryzyko związane z kredytami, przewidywać potencjalne straty i wykrywać nieuczciwe działania.

##### Przykład: Modelowanie ryzyka kredytowego

Jednym z kluczowych zastosowań uczenia maszynowego w finansach jest modelowanie ryzyka kredytowego, gdzie instytucje finansowe przewidują prawdopodobieństwo, że klient nie spłaci swojego kredytu. W tym celu można używać algorytmów takich jak regresja logistyczna, Random Forest czy XGBoost, które pomagają w analizie cech klienta i przewidywaniu ryzyka.

**Przykładowy kod dla modelu ryzyka kredytowego z użyciem Random Forest:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Wczytywanie danych o kredytobiorcach
data = pd.read_csv('credit_data.csv')

# Przygotowanie danych
X = data.drop('default', axis=1)  # cechy klienta, np. dochód, wiek, historia kredytowa
y = data['default']  # czy klient zbankrutował (1 - tak, 0 - nie)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
```

Model Random Forest może skutecznie wykrywać klientów o wysokim ryzyku niewypłacalności, co pomaga w podejmowaniu decyzji kredytowych.

##### Wykrywanie oszustw

Innym ważnym zastosowaniem uczenia maszynowego w finansach jest **wykrywanie oszustw**. Algorytmy takie jak Isolation Forest czy DBSCAN mogą być używane do wykrywania nietypowych transakcji, które mogą wskazywać na oszustwa.

**Przykładowy kod dla wykrywania oszustw z użyciem Isolation Forest:**

```python
from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.metrics import classification_report

# Wczytywanie danych transakcyjnych
data = pd.read_csv('transactions.csv')

# Przygotowanie danych (bez etykiet - nienadzorowane uczenie)
X = data.drop('is_fraud', axis=1)

# Trening modelu Isolation Forest
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X)

# Predykcje (1 - oszustwo, -1 - brak oszustwa)
y_pred = model.predict(X)
y_pred = [1 if x == -1 else 0 for x in y_pred]

# Ocena modelu
y_true = data['is_fraud']
print(classification_report(y_true, y_pred))
```

Wykrywanie oszustw za pomocą nienadzorowanego uczenia, takiego jak Isolation Forest, pozwala na identyfikację nietypowych transakcji bez potrzeby posiadania dużego zbioru danych oznaczonych.

#### 9.2. Diagnostyka medyczna

W branży medycznej uczenie maszynowe odgrywa kluczową rolę w diagnostyce chorób, analizie obrazów medycznych oraz personalizacji terapii. Algorytmy mogą pomóc w szybszym i dokładniejszym diagnozowaniu chorób, identyfikacji wzorców w danych genetycznych i biomedycznych oraz prognozowaniu skuteczności leczenia.

##### Przykład: Wykrywanie raka na podstawie danych medycznych

Uczenie maszynowe może być wykorzystywane do analizowania danych pacjentów, takich jak wyniki badań krwi, cechy genetyczne, i przewidywania występowania chorób, takich jak rak. Do takich zadań dobrze nadają się algorytmy klasyfikacyjne, takie jak SVM, Random Forest czy Gradient Boosting.

**Przykładowy kod dla klasyfikacji raka piersi z użyciem SVM:**

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Wczytywanie danych o pacjentach
data = pd.read_csv('breast_cancer_data.csv')

# Przygotowanie danych
X = data.drop('diagnosis', axis=1)  # cechy, np. wielkość guza, gęstość tkanki
y = data['diagnosis']  # diagnoza (1 - rak, 0 - brak raka)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu SVM
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
```

Algorytmy takie jak SVM są często stosowane w diagnostyce medycznej, ponieważ są w stanie skutecznie klasyfikować dane o złożonych zależnościach, takich jak dane biometryczne pacjentów.

##### Personalizacja terapii

Uczenie maszynowe może również wspierać **personalizację terapii**, gdzie na podstawie historii medycznej pacjenta, cech genetycznych i odpowiedzi na poprzednie terapie algorytmy pomagają przewidzieć najbardziej efektywną metodę leczenia. Algorytmy regresyjne i metody ensemble, takie jak Gradient Boosting, są używane do prognozowania skuteczności terapii.

**Przykładowy kod dla personalizacji terapii z użyciem Gradient Boosting:**

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Wczytywanie danych pacjentów i wyników terapii
data = pd.read_csv('therapy_data.csv')

# Przygotowanie danych
X = data.drop('therapy_success', axis=1)  # cechy, np. wiek, rodzaj terapii
y = data['therapy_success']  # skuteczność terapii (np. zmniejszenie objawów)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu Gradient Boosting
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

W ten sposób algorytmy regresyjne pomagają w prognozowaniu skuteczności różnych terapii, co pozwala na bardziej spersonalizowane podejście do leczenia pacjentów.

#### 9.3. Predykcje w marketingu

Uczenie maszynowe w marketingu umożliwia firmom lepsze zrozumienie swoich klientów, przewidywanie ich zachowań zakupowych oraz optymalizację kampanii marketingowych. Algorytmy mogą analizować wzorce zakupowe, przewidywać odejścia klientów (churn) i segmentować klientów na podstawie ich preferencji.

##### Przykład: Przewidywanie odejścia klientów (churn prediction)

Firmy mogą korzystać z modeli uczenia maszynowego

 do przewidywania, które z ich klientów są najbardziej narażone na rezygnację z usług (churn). Modele takie jak XGBoost czy Random Forest są często używane do tego celu, ponieważ potrafią modelować złożone zależności w danych o klientach.

**Przykładowy kod dla przewidywania churn z użyciem XGBoost:**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Wczytywanie danych o klientach
data = pd.read_csv('customer_data.csv')

# Przygotowanie danych
X = data.drop('churn', axis=1)  # cechy, np. liczba transakcji, wartość zakupów
y = data['churn']  # czy klient odszedł (1 - tak, 0 - nie)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
```

Przewidywanie churn pozwala firmom na podejmowanie działań prewencyjnych, takich jak oferowanie promocji dla klientów narażonych na odejście.

##### Segmentacja klientów

Segmentacja klientów to kolejna ważna aplikacja uczenia maszynowego w marketingu. Klientów można grupować na podstawie ich zachowań zakupowych, demografii lub preferencji. Algorytmy takie jak K-means lub DBSCAN mogą pomóc w automatycznym wykrywaniu segmentów klientów.

**Przykładowy kod dla segmentacji klientów z użyciem K-means:**

```python
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Wczytywanie danych o klientach
data = pd.read_csv('customer_data.csv')

# Przygotowanie danych
X = data[['annual_income', 'spending_score']]  # Przykładowe cechy: dochód roczny i poziom wydatków

# Klasteryzacja K-means
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

# Wizualizacja klastrów
plt.scatter(X['annual_income'], X['spending_score'], c=clusters, cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Segmentacja klientów za pomocą K-means')
plt.show()
```

Segmentacja klientów pomaga firmom lepiej zrozumieć różne grupy użytkowników i dostosować strategie marketingowe do ich potrzeb.

#### 9.4. Algorytmy specyficzne dla danej branży: algorytmy dla danych sekwencyjnych, algorytmy przestrzenne

Niektóre branże wymagają zastosowania specyficznych algorytmów dostosowanych do rodzaju danych, takich jak dane sekwencyjne (np. w analizie czasowej) lub dane przestrzenne (np. w analizie geograficznej).

##### Algorytmy dla danych sekwencyjnych

Dane sekwencyjne są szczególnie ważne w branżach takich jak finanse, handel czy medycyna, gdzie często mamy do czynienia z danymi czasowymi. Modele takie jak LSTM (Long Short-Term Memory) i GRU (Gated Recurrent Unit) są często stosowane do analizy danych sekwencyjnych, jednak w kontekście tego rozdziału skupimy się na tradycyjnych modelach, takich jak **ARIMA** (Autoregressive Integrated Moving Average).

**Przykładowy kod dla modelu ARIMA w prognozowaniu finansowym:**

```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt

# Wczytywanie danych (np. ceny akcji)
data = pd.read_csv('stock_prices.csv', index_col='date', parse_dates=True)

# Przygotowanie danych
y = data['close_price']

# Trening modelu ARIMA
model = ARIMA(y, order=(5, 1, 0))  # Model ARIMA (5,1,0)
model_fit = model.fit()

# Predykcje
predictions = model_fit.forecast(steps=10)
print(predictions)

# Wizualizacja
plt.plot(y, label='Ceny akcji')
plt.plot(predictions, label='Prognozy', color='red')
plt.title('Prognozowanie cen akcji z użyciem ARIMA')
plt.legend()
plt.show()
```

##### Algorytmy przestrzenne

Dane przestrzenne są powszechnie używane w branżach takich jak urbanistyka, geografia czy analiza ryzyka. Modele takie jak **DBSCAN** mogą być stosowane do analizy grupowań przestrzennych, co jest szczególnie przydatne w geograficznych analizach danych.

**Przykładowy kod dla algorytmu DBSCAN w analizie przestrzennej:**

```python
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt

# Wczytywanie danych przestrzennych (np. lokalizacje sklepów)
data = pd.read_csv('locations.csv')

# Przygotowanie danych (kolumny: szerokość i długość geograficzna)
X = data[['latitude', 'longitude']]

# Klasteryzacja DBSCAN
db = DBSCAN(eps=0.1, min_samples=5)
clusters = db.fit_predict(X)

# Wizualizacja klastrów
plt.scatter(X['latitude'], X['longitude'], c=clusters, cmap='plasma')
plt.xlabel('Szerokość geograficzna')
plt.ylabel('Długość geograficzna')
plt.title('Analiza klastrów przestrzennych z DBSCAN')
plt.show()
```

Algorytmy przestrzenne pomagają analizować dane geograficzne, co jest kluczowe w logistyce, analizach demograficznych i urbanistyce.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone