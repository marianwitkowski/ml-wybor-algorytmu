### 8. Hybrydyzacja algorytmów i nowoczesne podejścia

Hybrydyzacja algorytmów i nowoczesne podejścia w uczeniu maszynowym mają na celu łączenie różnych technik, aby uzyskać bardziej efektywne i dokładne modele. W tej części omówimy hybrydowe modele uczenia maszynowego, podejścia semi-nadzorowane do pracy z ograniczonymi danymi oraz przyszłość hybrydowych technik modelowania. Przykłady obejmują zaawansowane metody, które łączą różne algorytmy bez użycia sieci neuronowych, pozwalają na lepsze modelowanie przy ograniczonych danych, i wskazują na nowe kierunki rozwoju w tej dziedzinie.

#### 8.1. Hybrydowe modele uczenia maszynowego (połączenia różnych algorytmów, bez sieci neuronowych)

Hybrydowe modele uczenia maszynowego łączą różne algorytmy w celu uzyskania bardziej wszechstronnych i precyzyjnych modeli. Te podejścia polegają na tym, że każdy algorytm ma swoje mocne i słabe strony, a ich kombinacja może prowadzić do lepszych wyników. Często łączone są algorytmy klasyfikacyjne, regresyjne i zespołowe (ensemble).

##### Przykład: Połączenie k-NN z Random Forest

**k-Nearest Neighbors (k-NN)** to algorytm, który dobrze radzi sobie z małymi zbiorami danych, ale jest wrażliwy na szum i może być niewydajny dla dużych zbiorów danych. Z kolei **Random Forest** jest bardziej odporny na szum i dobrze działa w przypadku dużych zbiorów danych. Możemy połączyć te dwa podejścia, używając k-NN do wstępnej klasyfikacji, a Random Forest do ostatecznej decyzji.

**Przykładowy kod hybrydowego modelu k-NN + Random Forest:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Trening modelu k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Przewidywanie wyników k-NN na zbiorze testowym
knn_pred = knn.predict(X_test)

# Trening modelu Random Forest na wynikach k-NN
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, knn_pred)

# Predykcje
final_pred = rf.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, final_pred)
print(f'Hybrydowy model k-NN + Random Forest Accuracy: {accuracy}')
```

Powyższy przykład pokazuje, jak wyniki k-NN mogą być użyte jako dodatkowa warstwa wejściowa do modelu Random Forest, co prowadzi do poprawy dokładności.

##### Przykład: Połączenie SVM z Boostingiem

Można także połączyć Support Vector Machines (SVM) z algorytmem **Gradient Boosting**, aby uzyskać model bardziej odporny na dane o wysokiej złożoności. SVM może działać jako model bazowy, a Gradient Boosting może poprawiać wyniki przez iteracyjne korygowanie błędów.

**Przykładowy kod hybrydowego modelu SVM + Gradient Boosting:**

```python
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Trening modelu SVM
svm = SVC(probability=True, kernel='linear')
svm.fit(X_train, y_train)

# Przewidywanie wyników SVM na zbiorze testowym
svm_pred = svm.predict_proba(X_test)[:, 1]

# Trening modelu Gradient Boosting na wynikach SVM
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

# Predykcje
final_pred = gb.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, final_pred)
print(f'Hybrydowy model SVM + Gradient Boosting Accuracy: {accuracy}')
```

W tym przypadku Gradient Boosting działa jako metoda poprawy błędów generowanych przez SVM, prowadząc do lepszej ogólnej wydajności.

#### 8.2. Uczenie maszynowe z ograniczonymi danymi: podejścia semi-nadzorowane

**Uczenie semi-nadzorowane** jest podejściem, które łączy cechy uczenia nadzorowanego i nienadzorowanego, pozwalając na efektywne uczenie się przy ograniczonej liczbie oznakowanych danych. W rzeczywistości, w wielu przypadkach dostęp do oznakowanych danych może być ograniczony ze względu na wysokie koszty anotacji, natomiast nieoznaczone dane mogą być dostępne w dużych ilościach.

##### Przykład: Algorytm semi-nadzorowany Label Propagation

Jednym z podejść semi-nadzorowanych jest **Label Propagation**, gdzie etykiety są propagowane od znanych do nieznanych danych na podstawie podobieństw między nimi. To podejście pozwala na efektywne wykorzystanie nieoznaczonych danych do poprawy wyników modelu.

**Przykładowy kod dla Label Propagation:**

```python
from sklearn.semi_supervised import LabelPropagation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Ograniczenie liczby etykiet (tylko 30% danych jest oznakowanych)
random_unlabeled_points = np.random.rand(len(y_train)) < 0.7
y_train[random_unlabeled_points] = -1

# Trening modelu Label Propagation
label_prop_model = LabelPropagation()
label_prop_model.fit(X_train, y_train)

# Predykcje
y_pred = label_prop_model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Label Propagation Accuracy (semi-nadzorowane uczenie): {accuracy}')
```

Label Propagation pomaga przewidywać brakujące etykiety na podstawie wzorców w danych, co pozwala na efektywne wykorzystanie nieoznaczonych próbek.

##### Przykład: Algorytm semi-nadzorowany Self-training

**Self-training** to kolejne popularne podejście semi-nadzorowane, w którym model trenuje się na oznakowanych danych, a następnie stosuje się go do przewidywania etykiet dla nieoznaczonych danych. Model jest ponownie trenowany na wszystkich danych, w tym na nowych, przewidywanych etykietach.

**Przykładowy kod dla Self-training:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Ograniczenie liczby etykiet (tylko 30% danych jest oznakowanych)
random_unlabeled_points = np.random.rand(len(y_train)) < 0.7
y_train[random_unlabeled_points] = -1

# Trening modelu Self-training
base_classifier = RandomForestClassifier(random_state=42)
self_training_model = SelfTrainingClassifier(base_classifier)
self_training_model.fit(X_train, y_train)

# Predykcje
y_pred = self_training_model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Self-training Accuracy: {accuracy}')
```

Self-training jest skuteczną metodą, gdy dysponujemy dużą ilością nieoznaczonych danych, które można wykorzystać do wzmocnienia wydajności modelu.

#### 8.3. Rozwój i przyszłość hybrydowych technik modelowania

Hybrydyzacja algorytmów ma ogromny potencjał w przyszłości uczenia maszynowego, szczególnie w kontekście coraz większej złożoności problemów i róż

norodności danych. 

##### Hybrydyzacja z algorytmami ensemble i semi-nadzorowanymi

Kombinowanie technik uczenia nadzorowanego, nienadzorowanego oraz semi-nadzorowanego jest kolejnym etapem w rozwoju hybrydowych modeli. Algorytmy zespołowe, takie jak XGBoost, LightGBM czy CatBoost, można wzmacniać poprzez hybrydowe podejścia semi-nadzorowane, które pomagają wykorzystać potencjał nieoznaczonych danych.

**Przykładowy kod dla hybrydyzacji z XGBoost i Self-training:**

```python
import xgboost as xgb
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np

# Wczytywanie danych
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Ograniczenie liczby etykiet (tylko 30% danych jest oznakowanych)
random_unlabeled_points = np.random.rand(len(y_train)) < 0.7
y_train[random_unlabeled_points] = -1

# Trening modelu Self-training z XGBoost
base_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
self_training_model = SelfTrainingClassifier(base_classifier)
self_training_model.fit(X_train, y_train)

# Predykcje
y_pred = self_training_model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Hybrydowy model Self-training + XGBoost Accuracy: {accuracy}')
```

W tym przykładzie hybrydowy model semi-nadzorowany korzysta z mocy XGBoost i uczenia na danych nieoznaczonych, co może prowadzić do znacznej poprawy wyników, szczególnie przy ograniczonych zasobach danych oznaczonych.

##### Przyszłość hybrydowych technik

W przyszłości rozwój hybrydowych modeli będzie obejmował większe wykorzystanie podejść multi-modalnych, łączących różne typy danych, np. dane tekstowe, obrazy i liczby. Możliwe jest także większe zintegrowanie uczenia semi-nadzorowanego z technikami meta-algorytmicznymi, takimi jak automatyzacja tuningu hiperparametrów w oparciu o hybrydowe zestawy algorytmów.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone
