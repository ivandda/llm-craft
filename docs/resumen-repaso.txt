Métricas
Métricas


Validez / Plausibilidad
La salida es una combinación razonable?
Una respuesta novedosa pero absurda no es creativa.
Se puede medir con un LLM judge


Integridad de la entrada
La salida usa ambos conceptos de entrada?
La salida debería combinar ambos inputs, no dar sinónimos de una sola.
Se puede medir con un LLM judge
	

Novedad
La salida es diferente de outputs de train? Puede producir cosas “nuevas”?
No nos interesa que aprenda de memoria las recetas
Embeddings
	

Diversidad
¿El modelo genera outputs variados para la misma entrada?
Es interesante que pueda razonar cosas diferentes, pero parece ser la menos importantes.
Embeddings
	

Fidelidad al teacher
Los resultados son parecidos a los del teacher?
	



















Datasets
Datasets
Datasets que tenemos
* https://huggingface.co/datasets/ericlewis/infinite-craft-recipes
* https://github.com/R74nCom/Elementia-Dataset
* https://github.com/expitau/InfiniteCraftWiki
* https://github.com/redfast00/element-alchemy-cheater


Datasets que genearmos


El teacher genera algo del estilo:


A = fire
B = water


chosen output: steam
alternative valid outputs: mist, hot spring, sauna
near negative: smoke
easy negative: banana
rationale: fire heats water, producing steam
	



Programáticamente armamos:


{
  "input_a": "fire",
  "input_b": "water",
  "gold_or_teacher_output": "steam",
  "valid_alternatives": ["mist", "hot spring", "sauna"],
  "near_negatives": ["smoke"],
  "easy_negatives": ["banana"],
  "rationale": "Fire heats water, producing steam.",
  "preference_pairs": [
    ("steam", "smoke"),
    ("steam", "banana"),
    ("hot spring", "banana")
  ]
}
	



Experimentos




	Base student
	

	SFT recipes
	

	SFT razonamiento
	

	SFT + DPO
	

	SFT + RAG
	

	

	

Tab 4
Tiene que estar cerca del espacio latente, pero infrecuente. 
Podemos hacer una pre selección y depues hacer un reranking