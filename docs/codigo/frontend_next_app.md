# Frontend Next.js

La aplicacion jugable vive en `apps/web` y esta implementada como una app Next.js. El combinador consulta recetas en Postgres y, cuando no existe una receta para el par, genera una salida con Vertex AI y la guarda para reutilizarla.

## Setup local

Desde la raiz del repositorio:

```bash
cd apps/web
npm ci
npm run dev
```

La app queda disponible en `http://localhost:3000`. Si el puerto esta ocupado:

```bash
npm run dev -- -p 3001
```

Para habilitar generaciones nuevas con Vertex, copiar el ejemplo de entorno y cargar la key:

```bash
cp apps/web/.env.example apps/web/.env.local
```

Variables relevantes:

```text
VERTEX_API_KEY=...
VERTEX_MODEL=gemini-2.5-flash
DATABASE_URL=postgres://llm_craft:llm_craft_dev@localhost:5432/llm_craft
```

## Credenciales y modos

El backend mock mantiene usuarios en memoria y trae un usuario seeded:

```text
usuario: admin
password: admin
```

La UI incluye tres modos:

* `Sandbox`: inventario inicial amplio para explorar combinaciones.
* `Goal`: genera una meta alcanzable desde recetas reales. El selector de profundidad controla cuantas combinaciones tiene el plan validado antes de llegar al objetivo. El inventario inicial se elige automaticamente desde presets curados de 2 a 4 elementos.
* `Agent Test`: recibe la misma profundidad que `Goal`, genera una meta nueva con seed por corrida y ejecuta un agente con Vertex. El agente solo puede elegir pares del inventario; la app llama al combinador como tool y corta cuando alcanza el objetivo o usa 20 mezclas.

Dentro de `Goal`, el panel derecho permite reiniciar la meta actual o generar una nueva sin volver al menu principal. Generar una nueva meta reemplaza el objetivo, resetea inventario/tablero/historial y carga el leaderboard del nuevo `goalId`.

`Sandbox` y `Goal` tienen un selector `Combiner model` con `Gemini 2.5 Flash`, `Gemini 2.5 Pro` y `Gemini 2.5 Flash Lite`. La seleccion se persiste por usuario y modo, y se envia a `POST /api/combine`. El modelo solo se usa cuando el par no existe en `final-10k`: las recetas conocidas siguen ganando siempre para conservar determinismo. Las generaciones nuevas se guardan en datasets por modelo (`web-generated-gemini-2.5-pro`, por ejemplo); para el modelo default tambien se consulta el cache legacy `web-generated`.

El juego tambien incluye `DPO test mode`. Cuando esta activo y una combinacion tiene dos o mas salidas candidatas reales, la UI muestra dos o tres opciones, usa la eleccion como resultado descubierto y guarda el evento en Postgres para entrenamiento futuro.

Los cambios de perfil, logros destacados, leaderboard, preferencias DPO y recetas generadas se guardan en Postgres mediante endpoints tipados.

## Endpoints

Los endpoints viven bajo `apps/web/app/api`:

* `POST /api/auth/login`
* `POST /api/auth/register`
* `POST /api/auth/logout`
* `GET /api/auth/me`
* `POST /api/combine`
* `POST /api/goals/random`
* `POST /api/agent-test/run`
* `POST /api/dpo/preferences`
* `GET /api/leaderboard`
* `POST /api/leaderboard`
* `PATCH /api/profile`

Los tipos compartidos estan en `apps/web/src/lib/types.ts`. La combinacion de elementos usa `recipe_pairs` y `recipe_candidates`: primero busca en `final-10k`, luego en el dataset generado del modelo elegido y, solo para `gemini-2.5-flash`, en `web-generated` legacy. Si no encuentra el par, llama a Vertex desde el servidor, valida JSON y persiste la salida nueva en el dataset generado de ese modelo. Las metas aleatorias usan recetas importadas de `final-10k`.

## Validacion

```bash
cd apps/web
npm run typecheck
npm run test
npm run build
```

Para una prueba manual minima:

1. Ejecutar `npm run dev`.
2. Abrir `http://localhost:3000`.
3. Iniciar sesion con `admin/admin`.
4. Probar una combinacion conocida, por ejemplo `water` + `fire`.
5. Probar una combinacion no conocida para validar la generacion con Vertex y su persistencia.
6. Cambiar entre `Sandbox` y `Goal`, elegir profundidad y revisar el leaderboard.
7. En `Goal`, usar `Reset` para reiniciar la meta actual y `New goal` para generar otra meta desde cero.
8. Entrar a `Agent Test`, elegir profundidad 2 y revisar que el limite del reporte sea 20.
9. Activar `DPO test mode`, combinar un par con alternativas reales y elegir una salida.

## Integracion futura con SFT

La app todavia no invoca los scripts ni modelos SFT de `src/sft`. `POST /api/combine` ya conserva el contrato `CombineRequest` / `CombineResponse`, por lo que el punto de integracion esperado es reemplazar el cliente Vertex server-side por un servicio de inferencia propio cuando el modelo estudiante este listo.
