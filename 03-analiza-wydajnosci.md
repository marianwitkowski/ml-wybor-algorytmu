### 3. Analiza wydajności algorytmów w kontekście danych

Wybór odpowiedniego algorytmu uczenia maszynowego jest kluczowy dla osiągnięcia wysokiej jakości wyników, ale skuteczność algorytmu jest silnie uzależniona od charakterystyki danych, na których jest stosowany. W tym punkcie omówimy, jak wielkość i zróżnicowanie zbiorów danych wpływa na wybór algorytmu, jak radzić sobie z brakującymi wartościami w danych oraz jak zarządzać danymi o wysokiej wymiarowości za pomocą technik redukcji wymiarowości takich jak PCA (Principal Component Analysis) i LDA (Linear Discriminant Analysis). 

#### 3.1. Wpływ wielkości i zróżnicowania zbiorów danych na wybór algorytmu

Jednym z najważniejszych aspektów, które należy wziąć pod uwagę podczas wyboru algorytmu, jest rozmiar i zróżnicowanie danych. Różne algorytmy radzą sobie różnie w zależności od liczby próbek oraz liczby cech (atrybutów) w danych.

##### Małe zbiory danych

W przypadku małych zbiorów danych, bardziej skomplikowane algorytmy, takie jak maszyny wektorów nośnych (SVM) czy gradient boosting, mogą nie być najlepszym wyborem, ponieważ istnieje ryzyko przeuczenia (overfitting). W takich przypadkach często lepiej sprawdzają się prostsze algorytmy, które mają mniejszą tendencję do dopasowywania się do szumu w danych.

**Przykład: Regresja logistyczna**

Regresja logistyczna jest prostym i efektywnym algorytmem, który dobrze sprawdza się przy małych zbiorach danych. Jej przewaga polega na tym, że jest mniej podatna na przeuczenie w porównaniu do bardziej złożonych algorytmów.

**Przykładowy kod dla regresji logistycznej:**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('small_dataset.csv')

# Przygotowanie danych
X = data.drop('target', axis=1)
y = data['target']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu
model = LogisticRegression()
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

##### Duże zbiory danych

W przypadku dużych zbiorów danych, prostsze algorytmy mogą nie być w stanie skutecznie przetworzyć i wykorzystać wszystkich dostępnych informacji. W takich przypadkach warto sięgnąć po bardziej złożone algorytmy, takie jak lasy losowe (Random Forest) czy XGBoost, które lepiej radzą sobie z dużymi ilościami danych i mogą uchwycić bardziej skomplikowane wzorce.

**Przykład: Random Forest**

Random Forest jest jednym z najczęściej stosowanych algorytmów dla dużych zbiorów danych. Jego zaletą jest to, że składa się z wielu drzew decyzyjnych, co pozwala na lepsze generalizowanie i redukcję ryzyka przeuczenia.

**Przykładowy kod dla Random Forest:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('large_dataset.csv')

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
print(f'Accuracy: {accuracy}')
```

##### Zróżnicowanie danych

Różnorodność danych (zróżnicowanie cech i etykiet) może również wpływać na wybór algorytmu. W przypadku danych o dużym zróżnicowaniu, algorytmy, które potrafią uwzględniać złożone zależności między cechami, jak XGBoost czy SVM, mogą okazać się bardziej efektywne.

**Przykład: XGBoost**

XGBoost jest znany ze swojej zdolności do radzenia sobie z złożonymi i zróżnicowanymi danymi, dzięki czemu często przewyższa inne algorytmy w zadaniach związanych z klasyfikacją i regresją.

**Przykładowy kod dla XGBoost:**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('diverse_dataset.csv')

# Przygotowanie danych
X = data.drop('target', axis=1)
y = data['target']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

#### 3.2. Przetwarzanie danych z brakującymi wartościami i ich wpływ na algorytm

Brakujące dane są powszechnym problemem w rzeczywistych zbiorach danych i mogą znacząco wpłynąć na wydajność modelu. Istnieje kilka podejść do radzenia sobie z brakującymi wartościami, a wybór odpowiedniej metody może wpływać na decyzję dotyczącą algorytmu.

##### Usuwanie brakujących danych

Najprostszą metodą radzenia sobie z brakującymi danymi jest usunięcie wszystkich wierszy (próbek) lub kolumn, które zawierają brakujące wartości. Jest to jednak podejście, które może prowadzić do utraty cennych informacji, zwłaszcza w przypadku dużej liczby brakujących danych.

**Przykład: Usuwanie brakujących danych**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Wczytywanie danych
data = pd.read_csv('data_with_missing_values.csv')

# Usuwanie wierszy z brakującymi wartościami
data_clean = data.dropna()

# Przygotowanie danych
X = data_clean.drop('target', axis=1)
y = data_clean['target']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

##### Uzupełnianie brakujących danych

Innym podejściem jest uzupełnianie brakujących danych. Można to zrobić za pomocą różnych metod, takich jak zastąpienie brakujących wartości średnią, medianą, najczęściej występującą wartością (dla danych kategorycznych) lub bardziej zaawansowanymi metodami, jak KNN-imputation czy modele regresyjne.

**Przykład: Uzupełnianie brakujących danych przy użyciu średniej**

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Wczytywanie danych
data = pd.read_csv('data_with_missing_values.csv')

# Uzupełnianie brakujących wartości średnią
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(data.drop('target', axis=1))
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
print(f'Accuracy: {accuracy}')
```

##### Algorytmy odporne na brakujące dane

Niektóre algorytmy, takie jak drzewa decyzyjne i ich pochodne, są naturalnie odporne na brakujące dane. W takich przypadkach algorytm automatycznie uwzględnia brakujące wartości podczas podejmowania decyzji na poszczególnych węzłach drzewa, co może prowadzić do większej dokładności bez konieczności wstępnego przetwarzania danych.

**Przykład: Random Forest**

Jak wspomniano wcześniej, Random Forest jest algorytmem, który potrafi radzić sobie z brakującymi wartościami bez potrzeby ich wstępnego uzupełniania.

#### 3.3. Radzenie sobie z danymi o wysokiej wymiarowości: PCA, LDA i inne techniki redukcji wymiarowości

Wysoka wymiarowość danych, czyli duża liczba cech (atrybutów), może być problematyczna w uczeniu maszynowym. Duża liczba cech może prowadzić do "klątwy wymiarowości", gdzie algorytmy tracą zdolność do efektywnego generalizowania i stają się podatne na przeuczenie. W takich przypadkach stosuje się techniki redukcji wymiarowości, które pomagają zmniejszyć liczbę cech, zachowując jak najwięcej istotnych informacji.

##### Principal Component Analysis (PCA)

PCA to jedna z najpopularniejszych technik redukcji wymiarowości. Działa poprzez przekształcenie oryginalnych cech w nowy zestaw cech, zwanych głównymi składowymi (principal components), które są liniowymi kombinacjami oryginalnych cech. Pierwszych kilka głównych składowych zazwyczaj zawiera większość informacji z oryginalnego zestawu danych.

**Przykład: Redukcja wymiarowości za pomocą PCA**

```python
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('high_dimensional_dataset.csv')

# Przygotowanie danych
X = data.drop('target', axis=1)
y = data['target']

# Redukcja wymiarowości za pomocą PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Trening modelu
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

##### Linear Discriminant Analysis (LDA)

LDA to technika redukcji wymiarowości, która różni się od PCA tym, że uwzględnia klasy etykiet w danych. LDA dąży do maksymalizacji separacji między różnymi klasami, co czyni ją szczególnie użyteczną w zadaniach klasyfikacyjnych.

**Przykład: Redukcja wymiarowości za pomocą LDA**

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('high_dimensional_dataset.csv')

# Przygotowanie danych
X = data.drop('target', axis=1)
y = data['target']

# Redukcja wymiarowości za pomocą LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.3, random_state=42)

# Trening modelu
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

##### Inne techniki redukcji wymiarowości

Oprócz PCA i LDA istnieją inne techniki redukcji wymiarowości, takie jak **t-SNE** (t-Distributed Stochastic Neighbor Embedding) i **UMAP** (Uniform Manifold Approximation and Projection), które są szczególnie użyteczne w wizualizacji danych o wysokiej wymiarowości. 

**Przykład: Redukcja wymiarowości za pomocą t-SNE**

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('high_dimensional_dataset.csv')

# Przygotowanie danych
X = data.drop('target', axis=1)
y = data['target']

# Redukcja wymiarowości za pomocą t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Wizualizacja
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.colorbar()
plt.title('Wizualizacja t-SNE')
plt.show()
```

t-SNE jest szczególnie przydatne do wizualizacji danych o wysokiej wymiarowości w 2D lub 3D, co pozwala na lepsze zrozumienie struktury danych i związków między próbkami.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone