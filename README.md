
# Wybór algorytmu uczenia maszynowego

## Niniejsza publikacja przybliża temat związany z wyborem algorytmu uczenia maszynowego z uwzględnieniem specyfiki danych, wielkości zbioru czy analizowanego problemu.


#### Autor: Marian Witkowski marian.witkowski@gmail.com

---

### Spis Treści

1. **<a href='01-wstep.md'>Wstęp</a>**
   - Cel publikacji
   - Znaczenie wyboru odpowiedniego algorytmu w praktyce
   - Krótkie przypomnienie typów problemów (klasyfikacja, regresja, klasteryzacja) jako kontekst do dalszych rozważań

2. **<a href='02-kryteria-wyboru.md'>Zaawansowane Kryteria Wybierania Algorytmu</a>**
   - Złożoność obliczeniowa i skalowalność w dużych zbiorach danych
   - Równowaga między dokładnością a interpretowalnością
   - Zarządzanie trade-offami: Bias-Variance, Czas trenowania vs. Jakość modelu
   - Algorytmy zoptymalizowane pod kątem przetwarzania na sprzęcie GPU/TPU (bez omówienia sieci neuronowych)

3. **<a href='03-analiza-wydajnosci.md'>Analiza Wydajności Algorytmów w Kontekście Danych</a>**
   - Wpływ wielkości i zróżnicowania zbiorów danych na wybór algorytmu
   - Przetwarzanie danych z brakującymi wartościami i ich wpływ na algorytm
   - Radzenie sobie z danymi o wysokiej wymiarowości: PCA, LDA, i inne techniki redukcji wymiarowości

4. **<a href='04-zaawansowane-techniki.md'>Zaawansowane Techniki Walidacji i Testowania Modeli</a>**
   - Walidacja krzyżowa z dostosowanymi strategiami pod kątem specyfiki danych
   - Techniki resamplingu: Bootstrapping, Jackknife
   - Dobór optymalnych metryk oceny modelu dla konkretnych problemów

5. **<a href='05-przyklady.md'>Zaawansowane Przykłady Praktyczne</a>**
   - Wybór algorytmu dla problemu wieloklasowego (multi-class classification)
   - Algorytmy regresji w kontekście danych nieliniowych i nienormalnych rozkładów
   - Praktyczne aspekty algorytmów klasteryzacji dla dużych zbiorów danych (np. k-means++, DBSCAN)

6. **<a href='06-optymalizacja.md'>Optymalizacja i Tuning Algorytmów</a>**
   - Zaawansowane techniki Grid Search, Random Search, i Bayesian Optimization
   - Znaczenie regularyzacji i normalizacji: L1, L2, Elastic Net
   - Techniki automatyzacji procesu tuningu hiperparametrów

7. **<a href='07-ensemble-learn.md'>Ensemble Learning i Meta-algorytmy</a>**
   - Głębsze spojrzenie na metody zespołowe (Random Forest, Gradient Boosting, XGBoost, LightGBM)
   - Metaanaliza wyników z różnych algorytmów: Stacking, Blending
   - Optymalizacja modeli zespołowych pod kątem szybkości i wydajności

8. **<a href='08-hybrydyzacja.md'>Hybrydyzacja Algorytmów i Nowoczesne Podejścia</a>**
   - Hybrydowe modele uczenia maszynowego (połączenia różnych algorytmów, bez sieci neuronowych)
   - Uczenie maszynowe z ograniczonymi danymi: podejścia semi-nadzorowane
   - Rozwój i przyszłość hybrydowych technik modelowania

9. **<a href='09-zastosowanie.md'>Przypadki Zastosowań w Różnych Branżach</a>**
   - Analiza przypadków: Finansowe modele ryzyka, diagnostyka medyczna, predykcje w marketingu
   - Algorytmy specyficzne dla danej branży: algorytmy dla danych sekwencyjnych, algorytmy przestrzenne

10. **<a href='10-przypadki.md'>Przyszłość Algorytmów Uczenia Maszynowego</a>**
    - Nowe trendy: AutoML, federacyjne uczenie maszynowe, modele generatywne (bez sieci neuronowych)
    - Potencjał rozwoju i zastosowania najnowszych algorytmów w przyszłości

11. **<a href='11-bibliografia.md'>Bibliografia</a>**
