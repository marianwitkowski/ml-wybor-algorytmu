### 7. Ensemble learning i meta-algorytmy

**Ensemble Learning** (uczenie zespołowe) to zaawansowana technika w uczeniu maszynowym, która łączy moc wielu modeli w celu poprawy ogólnej wydajności systemu predykcyjnego. Zamiast polegać na jednym modelu, metody zespołowe wykorzystują wiele algorytmów lub instancji tego samego algorytmu, aby uzyskać bardziej stabilne i dokładne wyniki. W tej sekcji omówimy głębsze techniki ensemble learningu, takie jak Random Forest, Gradient Boosting, XGBoost i LightGBM. Przedstawimy również metaanalizę wyników z różnych algorytmów (Stacking, Blending) oraz optymalizację modeli zespołowych pod kątem szybkości i wydajności.

#### 7.1. Głębsze spojrzenie na metody zespołowe (Random Forest, Gradient Boosting, XGBoost, LightGBM)

##### Random Forest

**Random Forest** to klasyczny algorytm zespołowy, który tworzy wiele drzew decyzyjnych, a następnie łączy ich wyniki w celu uzyskania bardziej dokładnych prognoz. Każde drzewo w Random Forest jest trenowane na losowym podzestawie danych, co pomaga zredukować wariancję i poprawić zdolność modelu do generalizacji. Algorytm ten jest szczególnie skuteczny w przypadkach, gdy dane są szumowe lub mają wiele wymiarów.

**Przykładowy kod dla Random Forest:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Trening modelu Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

Random Forest jest potężnym narzędziem, które wykorzystuje prostotę drzew decyzyjnych, ale dzięki agregacji wyników z wielu drzew osiąga znacznie lepsze wyniki niż pojedyncze drzewo decyzyjne.

##### Gradient Boosting

**Gradient Boosting** to bardziej zaawansowana metoda zespołowa, która buduje model sekwencyjnie, dodając kolejne drzewa decyzyjne, gdzie każde kolejne drzewo stara się skorygować błędy poprzedniego modelu. W porównaniu do Random Forest, Gradient Boosting ma na celu zmniejszenie błędu poprzez stopniowe doskonalenie modelu, co czyni go bardziej precyzyjnym, ale także bardziej podatnym na przeuczenie (overfitting).

**Przykładowy kod dla Gradient Boosting:**

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Trening modelu Gradient Boosting
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

Gradient Boosting skutecznie poprawia dokładność predykcji, zwłaszcza w przypadkach, gdy dane mają skomplikowane wzorce. Jednakże, jego głównym wyzwaniem jest tendencja do przeuczenia, co wymaga dostrojenia hiperparametrów.

##### XGBoost

**XGBoost** (Extreme Gradient Boosting) to jedna z najpopularniejszych implementacji Gradient Boostingu, która jest zoptymalizowana pod kątem wydajności i szybkości. XGBoost wprowadza szereg optymalizacji, takich jak równoległość obliczeń, regularyzacja, a także obsługa brakujących danych, co czyni go jednym z najpotężniejszych algorytmów dostępnych w praktyce.

**Przykładowy kod dla XGBoost:**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Trening modelu XGBoost
model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

XGBoost jest szczególnie efektywny przy dużych zbiorach danych i w konkursach uczenia maszynowego, gdzie często przewyższa inne algorytmy.

##### LightGBM

**LightGBM** (Light Gradient Boosting Machine) to kolejny zaawansowany algorytm zespołowy, który został zaprojektowany do pracy z dużymi zbiorami danych. LightGBM jest szybszy niż XGBoost i lepiej radzi sobie z wysokowymiarowymi danymi, co czyni go idealnym wyborem w projektach, które wymagają wysokiej wydajności obliczeniowej.

**Przykładowy kod dla LightGBM:**

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Trening modelu LightGBM
model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

LightGBM jest zoptymalizowany pod kątem dużych zbiorów danych i oferuje doskonałą wydajność, nawet w przypadkach, gdy mamy do czynienia z wieloma wymiarami.

#### 7.2. Metaanaliza wyników z różnych algorytmów: Stacking, Blending

Kiedy różne modele dają różne wyniki, można je połączyć za pomocą technik **Stacking** lub **Blending**, aby uzyskać model o lepszej wydajności. 

##### Stacking

**Stacking** polega na trenowaniu wielu modeli bazowych (np. regresja logistyczna, SVM, Random Forest), a następnie na użyciu nowego modelu (tzw. model meta), który na podstawie wyników tych modeli bazowych dokonuje ostatecznych predykcji. Jest to potężna technika, która często prowadzi do lepszych wyników niż każdy z modeli bazowych osobno.

**Przykładowy kod dla Stacking:**

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Definiowanie modeli bazowych
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42))
]

# Stacking
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_model.fit(X_train, y_train)

# Predykcje
y_pred = stacking_model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

Stacking pozwala na wykorzystanie mocnych stron różnych algorytmów, aby uzyskać lepsze wyniki niż jakikolwiek pojedynczy model.

##### Blending

**Blending** to uproszczona wersja Stacking, w której używamy wyników kilku modeli bazowych, ale zamiast trenować model meta na całym zbiorze, modele

 bazowe są trenowane na jednej części danych, a ich wyniki są łączone w sposób liniowy lub nieliniowy na podstawie zbioru walidacyjnego.

**Przykładowy kod dla Blending:**

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Trening dwóch modeli
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lr_model = LogisticRegression()

rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Przewidywanie na zbiorze testowym
rf_pred = rf_model.predict_proba(X_test)[:, 1]
lr_pred = lr_model.predict_proba(X_test)[:, 1]

# Blending: średnia wyników
blended_pred = (rf_pred + lr_pred) / 2
blended_pred_class = np.where(blended_pred > 0.5, 1, 0)

# Ocena modelu
accuracy = accuracy_score(y_test, blended_pred_class)
print(f'Blended Accuracy: {accuracy}')
```

Blending jest prostszą techniką niż Stacking, ale często zapewnia bardzo dobre wyniki, zwłaszcza w przypadkach, gdy różne modele dają różne wyniki.

#### 7.3. Optymalizacja modeli zespołowych pod kątem szybkości i wydajności

Modele zespołowe mogą być bardzo dokładne, ale często są również kosztowne obliczeniowo, ponieważ angażują wiele modeli naraz. Aby zoptymalizować modele zespołowe pod kątem szybkości i wydajności, stosuje się różne techniki:

- **Early Stopping**: W metodach takich jak Gradient Boosting, można przerwać trenowanie modelu, gdy przestaje on poprawiać wyniki na zbiorze walidacyjnym.
- **Parallelizacja**: Random Forest i XGBoost mogą być trenowane równolegle, co pozwala na znaczne przyspieszenie procesu trenowania, zwłaszcza przy dużych zbiorach danych.
- **Pruning**: Usuwanie nieistotnych drzew decyzyjnych w modelach takich jak Random Forest może przyspieszyć predykcje bez znacznej utraty dokładności.

**Przykład optymalizacji modelu XGBoost za pomocą Early Stopping:**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Konwersja danych do formatu DMatrix (dla XGBoost)
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)

# Parametry modelu XGBoost
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 4,
    'eta': 0.3,
    'eval_metric': 'mlogloss'
}

# Trenowanie modelu z Early Stopping
watchlist = [(train_data, 'train'), (test_data, 'eval')]
model = xgb.train(params, train_data, num_boost_round=100, early_stopping_rounds=10, evals=watchlist)

# Predykcje
y_pred = model.predict(test_data)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

Early Stopping pozwala na zatrzymanie trenowania modelu, gdy poprawa na zbiorze walidacyjnym zatrzymuje się, co oszczędza czas i zasoby.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone