> ⚠️ **Documento histórico — ANTEPROYECTO (propuesta inicial).** Describe el plan
> original, no lo que finalmente se implementó. Varias piezas propuestas aquí **no se
> ejecutaron o cambiaron**: el dataset final **no tiene** `near_negatives`/`easy_negatives`
> ni `preference_pairs` (solo candidatos válidos `observed`/`teacher`); DPO se hizo
> **on-policy** (chosen = canónico corto, rejected = la propia salida verbosa/inválida del
> modelo, seleccionados por reglas), **no** con negativos del teacher; y **no** se
> implementaron RAG (M5), evaluación *goal-directed*, evaluación humana ni el *Teacher
> Fidelity Score* (TFS). El informe fiel a lo realizado es **`informe/latex/acl_latex.tex`**
> y `docs/resultados.md`. Ver correcciones en `docs/informe/correcciones_informe.md`.

# Destilación y evaluación de creatividad composicional en LLMs pequeños

**Santino Galliano**  
**Manuel Borrell**  
**Elias Goro Picart**  
**Lara Leporace**  
**Iván Domínguez de Álzaga**

`{sgalliano, mborrell, egoropicart, lleporace, idominguezdealzaga}@udesa.edu.ar`

Universidad de San Andrés

## Resumen

Este trabajo propone estudiar cómo distintas formas de supervisión sintética afectan la creatividad composicional de modelos de lenguaje pequeños. Usamos un entorno inspirado en *Infinite Craft*, donde dos conceptos se combinan para producir uno nuevo, por ejemplo `(fire, water) -> steam`. A diferencia de una tarea de clasificación cerrada, una misma combinación puede admitir múltiples respuestas plausibles. El proyecto compara modelos *student* entrenados con recetas, explicaciones, preferencias y recuperación de ejemplos, y separa dos dimensiones: creatividad composicional y fidelidad al *teacher*. La contribución esperada es un marco reproducible para evaluar si la destilación induce composición conceptual novedosa o simplemente imitación del modelo grande.

## 1. Introducción

La creatividad en modelos de lenguaje es difícil de evaluar porque una respuesta puede ser novedosa pero incoherente, correcta pero trivial, o sorprendente pero arbitraria. Esta dificultad se vuelve especialmente relevante cuando modelos pequeños son entrenados con datos generados por modelos grandes: una mejora aparente puede reflejar memorización o imitación del *teacher*, y no necesariamente una mayor capacidad de composición conceptual.

Proponemos estudiar este problema en una tarea inspirada en *Infinite Craft*, es un juego de combinación conceptual en el que el usuario parte de un conjunto pequeño de elementos básicos, como water, fire, earth y air, y combina pares de elementos para descubrir nuevos conceptos. Cada acción se modela como una receta local:

$$
x = (a, b), \qquad (a, b) \to y,
\tag{1}
$$

donde $a$ y $b$ son conceptos de entrada e $y$ es el concepto generado. Por ejemplo, `fire + water` puede producir steam, pero también alternativas plausibles como mist, hot spring o sauna. Por lo tanto, el objetivo no es predecir una única etiqueta correcta, sino generar salidas plausibles, novedosas y justificadas por ambos conceptos de entrada.

La pregunta central es: *¿qué tipo de supervisión sintética permite que un modelo pequeño pase de imitar recetas a componer conceptos de manera novedosa, plausible y útil?* No asumimos que el modelo grande sea creativo en sentido absoluto. Lo usamos como fuente escalable de recetas, explicaciones y preferencias, mientras que la creatividad del *student* se mide con una métrica externa. Esto permite analizar el posible *trade-off* entre fidelidad al *teacher* y exploración creativa propia.

## 2. Trabajo relacionado

El proyecto se apoya en cinco líneas de trabajo. Primero, la composicionalidad semántica estudia cómo el significado puede representarse y combinarse en espacios vectoriales, desde Word2Vec y GloVe hasta trabajos sobre analogías y composicionalidad aditiva (Mikolov et al., 2013; Pennington et al., 2014; Allen and Hospedales, 2019). Nuestro trabajo traslada esta pregunta a modelos generativos que producen nuevos conceptos a partir de pares de entrada.

Segundo, la destilación de conocimiento busca transferir capacidades de un modelo grande a uno pequeño (Hinton et al., 2015; Xu et al., 2024). En nuestro caso, el *teacher* no se usa como verdad absoluta, sino como generador de supervisión sintética. Tercero, la destilación con explicaciones o *rationales* permite entrenar modelos con señales intermedias que justifican una respuesta (Li et al., 2023). Esto es relevante porque una combinación válida debería explicar cómo ambos conceptos contribuyen al resultado.

Cuarto, los métodos de aprendizaje por preferencias, como DPO (Rafailov et al., 2023), son adecuados cuando existen múltiples respuestas posibles y no una única etiqueta correcta. Finalmente, los trabajos de evaluación de creatividad en LLMs muestran que la creatividad no puede reducirse a novedad: una salida debe ser original, pero también adecuada, específica y defendible. El gap que abordamos es la falta de evidencia sobre cómo distintas formas de supervisión sintética afectan la creatividad composicional de modelos pequeños en un entorno abierto, interactivo y con múltiples respuestas válidas.

## 3. Tarea y datos

La tarea base consiste en predecir o generar un concepto $y$ a partir de dos conceptos disponibles $(a, b)$. Partiremos de datasets públicos de recetas tipo *Infinite Craft* o *element alchemy*, y construiremos una versión enriquecida con un LLM *teacher*. Cada ejemplo incluirá campos como `input_elements`, `gold_result`, `teacher_candidates`, `rationale`, `score` y `label`.

Para `fire + water`, por ejemplo, el *teacher* podría generar steam como respuesta preferida, hot spring como alternativa plausible, smoke como negativo cercano porque usa principalmente fire, y banana como negativo fácil. Los negativos cercanos son importantes porque obligan al modelo a distinguir entre usar un solo input y componer ambos.

La evaluación no se limitará a un split aleatorio. Usaremos particiones que midan generalización composicional: recetas similares a entrenamiento, combinaciones nuevas con elementos vistos, conceptos raros o abstractos, casos adversariales donde una respuesta superficial usa solo un input, y tareas *goal-directed* donde el modelo debe alcanzar un objetivo mediante una secuencia de pasos válidos.

## 4. Métodos propuestos

Compararemos distintas variantes de entrenamiento y uso del modelo. M0 será el *teacher*, usado como generador de datos y no como ground truth absoluto. M1 será un *student* base sin ajuste específico. M2 usará SFT con pares simples $(A, B) \to C$. M3 usará SFT con respuesta y *rationale*, para evaluar si las explicaciones mejoran la integración de ambos inputs. M4 usará DPO con pares chosen/rejected, para estudiar si las preferencias mejoran plausibilidad y reducen respuestas absurdas. M5 usará RAG con recetas similares recuperadas como contexto, para comparar conocimiento internalizado contra memoria externa.

Estas variantes permiten formular hipótesis concretas. Esperamos que SFT mejore exact match pero pueda reducir diversidad; que los *rationales* mejoren la integración de inputs; que DPO aumente plausibilidad pero vuelva al modelo más conservador; y que RAG ayude especialmente en tareas *goal-directed*, donde recordar recetas previas puede facilitar la construcción de caminos.

## 5. Evaluación

La evaluación se organiza en cuatro niveles. En el nivel local, cada receta $(A, B) \to C$ se evalúa por plausibilidad, especificidad, novedad, integración de inputs y calidad del *rationale*. La integración es clave: `fire + water ->steam` usa ambos conceptos, mientras que `fire + water ->smoke` depende principalmente de fire e ignora water.

En el nivel de distribución, para una misma entrada el modelo genera $K$ respuestas. Esto permite medir diversidad y detectar colapso de modo. Por ejemplo, `{steam, vapor, water vapor}` tiene baja diversidad, mientras que `{steam, mist, hot spring, sauna}` explora alternativas semánticamente más distintas.

En el nivel *goal-directed*, el agente parte de elementos iniciales como water, fire, earth y air, y debe alcanzar un objetivo. Cada paso debe ser localmente válido: el objetivo final puede guiar la búsqueda, pero no justificar saltos inválidos como `earth + fire ->intelligence`. Mediremos tasa de éxito, tasa de caminos válidos, saltos inválidos al objetivo, longitud promedio del camino, llamadas al modelo, tokens, latencia y costo.

Finalmente, realizaremos una evaluación humana parcial sobre una muestra estratificada de recetas fáciles, abstractas, adversariales y casos donde los modelos discrepan. Esto servirá para comparar el juicio humano con el LLM-as-judge y estimar la confiabilidad de las métricas automáticas.

## 6. Especificación matemática

Sea $\mathcal{D}_{eval}$ el conjunto de recetas de evaluación. Para cada entrada $x = (a, b)$, un modelo $M_\theta$ genera un conjunto de $K$ respuestas:

$$
Y_\theta(x) = \{y_1, \ldots, y_K\}.
\tag{2}
$$

Definimos la creatividad composicional local como:

$$
C(x) = \alpha \frac{1}{K} \sum_{y \in Y_\theta(x)} q(x, y)^\lambda n(x, y) + (1 - \alpha)d(Y_\theta(x)),
\tag{3}
$$

donde $q(x, y)$ mide plausibilidad (se podría modelar con un LLM as a judge), $n(x, y)$ mide novedad, $d(Y_\theta(x))$ mide diversidad, $\alpha \in [0, 1]$ controla el peso entre calidad individual y diversidad, y $\lambda \geq 1$ penaliza respuestas poco plausibles.

El *Compositional Creativity Score* se obtiene promediando:

$$
\operatorname{CCS}(M_\theta) = \frac{1}{|\mathcal{D}_{eval}|} \sum_{x \in \mathcal{D}_{eval}} C(x).
\tag{4}
$$

La novedad puede estimarse como la distancia entre la salida generada y el vecino conocido más cercano dentro del conjunto de combinaciones observadas para el input $x$:

$$
n(x, y) = \min_{z \in \mathcal{Y}_{train}(x)} \delta(e(y), e(z)),
\tag{5}
$$

donde $e(\cdot)$ es un encoder semántico fijo, $\delta$ es una métrica de distancia semántica, por ejemplo la distancia coseno, y $\mathcal{Y}_{train}(x)$ representa las salidas conocidas asociadas al input $x$ en el conjunto de entrenamiento.

Otra forma posible de estimar la novedad es comparar directamente la salida generada con los dos conceptos de entrada. Si $x = (a, b)$ y $y = c$, se puede definir:

$$
n(x, y) = \frac{d_{\cos}(e(a), e(c)) + d_{\cos}(e(b), e(c))}{2},
\tag{6}
$$

donde $d_{\cos}$ mide la distancia coseno entre embeddings. Esta formulación captura qué tan alejada semánticamente está la salida $c$ respecto de los conceptos originales $a$ y $b$. La diversidad se calcula como distancia promedio entre outputs:

$$
d(Y_\theta(x)) = \frac{2}{K(K - 1)} \sum_{1 \leq r < t \leq K} \delta(e(y_r), e(y_t)).
\tag{7}
$$

Como análisis separado, medimos fidelidad al *teacher*:

$$
\operatorname{TFS}(M_S, M_T) = w_A A + w_R R + w_P P + w_G G,
\tag{8}
$$

donde $A$ mide acuerdo de outputs, $R$ acuerdo de rankings, $P$ acuerdo de preferencias y $G$ similitud entre explicaciones. Esta métrica no mide creatividad, sino imitación. La comparación entre CCS y TFS permitirá estudiar si mayor fidelidad al *teacher* implica, o no, mayor creatividad composicional.

## 7. Contribuciones esperadas y limitaciones

El proyecto busca aportar: un benchmark reproducible para evaluar composición conceptual generativa; una comparación controlada entre SFT, SFT con *rationales*, DPO y RAG; una métrica operacional de creatividad composicional; una métrica separada de fidelidad al *teacher*; y una plataforma interactiva para analizar recetas locales y trayectorias *goal-directed*.

La principal limitación es que no medimos creatividad humana en sentido psicológico o filosófico, sino una definición operacional. Además, las métricas basadas en embeddings (una opción simple y reproducible es usar modelos abiertos de Sentence Transformers, como all-MiniLM-L6-v2, que produce vectores densos útiles para similitud semántica, clustering y búsqueda semántica) y LLM-as-judge pueden introducir sesgos. Por eso, el sistema debería reportar el encoder usado, validar una muestra con evaluación humana y realizar análisis de error sobre casos de baja plausibilidad, baja diversidad, falta de integración de inputs y saltos inválidos hacia el objetivo.

## 8. Conclusión

Proponemos un marco experimental para estudiar si modelos pequeños pueden aprender creatividad composicional a partir de supervisión sintética. Al separar creatividad composicional de fidelidad al *teacher*, el proyecto permite distinguir entre modelos que simplemente imitan asociaciones del modelo grande y modelos que generan combinaciones nuevas, plausibles y defendibles. Esta distinción es central para entender qué se transfiere realmente cuando se destilan modelos grandes hacia modelos más pequeños.

## References

Carl Allen and Timothy M. Hospedales. 2019. Analogies explained: Towards understanding word embeddings. In *Proceedings of the 36th International Conference on Machine Learning*, pages 223-231.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*.

Liunian Harold Li, Jack Hessel, Youngjae Yu, Xiang Ren, Kai-Wei Chang, and Yejin Choi. 2023. Symbolic chain-of-thought distillation: Small models can also “think” step-by-step. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics*, pages 2665-2679.

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word representations in vector space. In *Proceedings of the International Conference on Learning Representations Workshop*.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global vectors for word representation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*, pages 1532-1543.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems*, 36:53728-53741.

Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Jinyang Li, Can Xu, Dacheng Tao, and Tianyi Zhou. 2024. A survey on knowledge distillation of large language models. *arXiv preprint arXiv:2402.13116*.
