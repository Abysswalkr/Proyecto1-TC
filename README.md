# Proyecto – Construcción y Simulación de Autómatas Finitos

## 📌 Descripción

Este proyecto implementa un motor de **expresiones regulares** en Python que permite:

1. **Conversión de ER infix → postfix** (algoritmo **Shunting Yard**).
2. **Construcción de AFN** mediante el **algoritmo de Thompson**.
3. **Conversión AFN → AFD** usando el método de **subconjuntos**.
4. **Minimización del AFD** con el algoritmo de **Hopcroft**.
5. **Simulación de cadenas** sobre AFN, AFD y AFD mínimo.
6. **Generación de gráficas PNG** de cada autómata.
7. **Verificación de equivalencia DFA ≡ DFAmin**.
8. **Pruebas aleatorias NFA ↔ DFA** para validar la construcción.

El programa soporta:

* Operadores: `*`, `+`, `?`, `|`, concatenación.
* Cuantificadores: `{m}`, `{m,}`, `{m,n}`.
* Punto `.` como **comodín** (cualquier símbolo).
* Clases de caracteres `[a-z]`, `[0-9]`, con rangos y escapes.

---

## ⚙️ Requisitos

* Python 3.9+
* Biblioteca `matplotlib` para graficar:

```bash
pip install matplotlib
```

---

## 🚀 Uso

Ejecuta el programa desde la terminal:

```bash
python AutomataEngine.py [opciones]
```

### Opciones principales

* `--mode`

  * `A` → Solo construye y simula AFN.
  * `B` → Construye AFN → AFD → AFDmin.
  * `ALL` → Ejecuta todo el pipeline.

* **Entrada de expresiones**:

  * `--regex "expresion"` → Procesa una única expresión.
  * `--input regexes.txt` → Procesa todas las expresiones de un archivo (una por línea).

* **Entrada de palabras**:

  * `--word cadena` → Usa la misma cadena para todas las ER.
  * `--words words.txt` → Archivo paralelo con cadenas (una por línea).

* **Otros parámetros útiles**:

  * `--outdir out_all` → Carpeta de salida para imágenes.
  * `--eps "ε"` → Cambia el símbolo visible de épsilon en las gráficas.
  * `--show-steps` → Muestra paso a paso el algoritmo Shunting Yard.
  * `--check` → Verifica formalmente la equivalencia `AFD ≡ AFDmin`.
  * `--random N` → Corre pruebas aleatorias NFA↔DFA con N cadenas.
  * `--alphabet "abc"` → Define alfabeto para las pruebas aleatorias.

---

## 📂 Ejemplos de ejecución

### 1) Procesar una sola expresión

```bash
python AutomataEngine.py --mode ALL --regex "(a|b)*abb(a|b)*" --word babbaaaa --outdir out_all --check --random 200 --alphabet ab
```

### 2) Procesar un archivo con varias expresiones

```bash
python AutomataEngine.py --mode ALL --input regexes.txt --word babbaaaa --outdir out_all --check --random 300 --alphabet ab
```

### 3) Probar distintas palabras por expresión

```bash
python AutomataEngine.py --mode ALL --input regexes.txt --words words.txt --outdir out_all --check --random 300 --alphabet ab
```

### 4) Mostrar pasos del algoritmo Shunting Yard

```bash
python AutomataEngine.py --mode A --regex "a(a|b*)b+a?" --show-steps
```

---

## 📊 Salida del programa

Por cada expresión procesada, el programa genera:

1. **Postfix canónica** de la expresión regular.
2. **Imágenes**:

   * `afn_i.png` → AFN por Thompson.
   * `afd_i.png` → AFD por subconjuntos.
   * `afd_min_i.png` → AFD minimizado.
3. **Simulación de la cadena w** en NFA, DFA y DFAmin (`sí` o `no`).


**Ejemplo de salida**:

```
[1] ER: (a|b)*abb(a|b)*
POSTFIX (canónica): a b | * a · b · b · a b | * ·
Imagen AFN -> out_all\afn_1.png
w = 'babbaaaa'  =>  NFA: sí
Imagen AFD     -> out_all\afd_1.png
Imagen AFDmin  -> out_all\afd_min_1.png
w = 'babbaaaa'  =>  NFA: sí | DFA: sí | DFAmin: sí
Equivalencia DFA ≡ DFAmin: OK
Prueba NFA↔DFA aleatoria: OK=300  MISMATCH=0  (N=300)
```

---

## ✅ Interpretación de resultados

* Si aparece **“sí”** → la cadena **pertenece** al lenguaje de la ER.
* Si aparece **“no”** → la cadena **no pertenece**.
* `Equivalencia DFA ≡ DFAmin: OK` → el minimizado reconoce exactamente el mismo lenguaje.
* `MISMATCH=0` → no hubo diferencias entre AFN y AFD en las pruebas aleatorias, validando la construcción.

---

## 📌 Créditos

Proyecto de **Teoría de la Computación**
Universidad del Valle de Guatemala – 2025

