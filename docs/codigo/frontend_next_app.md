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

## Sesiones y modos

El acceso es anonimo por defecto: al abrir la app, si no hay sesion, se crea automaticamente un usuario invitado (`guest-<id>`) via `POST /api/auth/guest` y se setea la cookie de sesion (httpOnly, un año de vigencia, `Secure` en produccion). Los invitados pueden pasar a una cuenta con nombre desde el boton `Sign in` del menu. Los usuarios se guardan en Postgres y existe un usuario seeded para administracion:

```text
usuario: admin
password: admin
```

Los endpoints que disparan inferencia (`/api/combine`, `/api/goals/random`, `/api/agent-test/run`) exigen sesion y tienen rate limiting por usuario y por IP respaldado en Postgres (`rate_limit_counters`, migracion `004`; politicas en `src/lib/server/rateLimit.ts`). La creacion de invitados tambien esta limitada por IP.

La UI incluye tres modos:

* `Sandbox`: inventario inicial amplio para explorar combinaciones.
* `Goal`: genera una meta alcanzable desde recetas reales. El selector de profundidad controla cuantas combinaciones tiene el plan validado antes de llegar al objetivo. El inventario inicial se elige automaticamente desde presets curados de 2 a 4 elementos.
* `Agent Test`: recibe la misma profundidad que `Goal`, genera una meta nueva con seed por corrida y ejecuta un agente con Vertex. El agente solo puede elegir pares del inventario; la app llama al combinador como tool y corta cuando alcanza el objetivo o usa 20 mezclas.

Dentro de `Goal`, el panel derecho permite reiniciar la meta actual o generar una nueva sin volver al menu principal. Generar una nueva meta reemplaza el objetivo, resetea inventario/tablero/historial y carga el leaderboard del nuevo `goalId`.

`Sandbox` y `Goal` tienen un selector `Combiner model` con `Gemini 2.5 Flash`, `Gemini 2.5 Pro` y `Gemini 2.5 Flash Lite`. La seleccion se persiste por usuario y modo, y se envia a `POST /api/combine`. El modelo solo se usa cuando el par no existe en `final-10k`: las recetas conocidas siguen ganando siempre para conservar determinismo. Las generaciones nuevas se guardan en datasets por modelo (`web-generated-gemini-2.5-pro`, por ejemplo); para el modelo default tambien se consulta el cache legacy `web-generated`.

El juego tambien incluye el toggle `Help train the AI` (captura de preferencias DPO), activado por defecto. Las rondas de preferencia son periodicas (combinaciones 3, 8, 13, ...): la UI pide candidatos a `POST /api/dpo/candidates`, que arma hasta 3 opciones "a ciegas" mezclando la salida canonica almacenada con generaciones en vivo de Qwen y Gemini (cada candidato lleva `generatedBy` en el evento guardado, pero la UI no muestra que modelo lo genero, para no sesgar). La opcion canonica siempre esta incluida (los goals se calculan con recetas rank-1). El jugador puede saltear con `Skip` (se usa la salida top y no se guarda preferencia). Los conceptos sin emoji reciben uno deterministico por nombre (`src/lib/emoji.ts`), que tambien define el tinte de color estable de cada ficha.

`Agent Test` funciona como arena de LLMs: cada corrida se persiste en `agent_runs` (migracion `005`) y la UI muestra un ranking por profundidad (tasa de exito, combinaciones promedio). El seed de la meta es diario por profundidad, asi todos los modelos enfrentan la misma meta y el ranking es comparable. Las metas aleatorias ahora tambien incluyen recetas generadas por modelos (`web-generated-*`); ante el mismo par, `final-10k` tiene prioridad.

Los cambios de perfil, logros destacados, leaderboard, preferencias DPO y recetas generadas se guardan en Postgres mediante endpoints tipados.

## Endpoints

Los endpoints viven bajo `apps/web/app/api`:

* `POST /api/auth/guest`
* `POST /api/auth/login`
* `POST /api/auth/register`
* `POST /api/auth/logout`
* `GET /api/auth/me`
* `POST /api/combine`
* `POST /api/goals/random`
* `GET /api/agent-test/runs` (feed publico del arena: goal del dia + corridas recientes)
* `GET /api/agent-test/rankings`
* `POST /api/admin/arena/run` (solo admin: correr un modelo contra el goal del dia)
* `POST /api/dpo/candidates`
* `POST /api/dpo/preferences`
* `GET /api/leaderboard`
* `POST /api/leaderboard`
* `PATCH /api/profile`
* `GET /api/admin/vm` / `POST /api/admin/vm` (solo admin)

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
2. Abrir `http://localhost:3000` y verificar que se crea una sesion de invitado automaticamente.
3. (Opcional) Iniciar sesion con `admin/admin` desde `Sign in`.
4. Probar una combinacion conocida, por ejemplo `water` + `fire`.
5. Probar una combinacion no conocida para validar la generacion con Vertex y su persistencia.
6. Cambiar entre `Sandbox` y `Goal`, elegir profundidad y revisar el leaderboard.
7. En `Goal`, usar `Reset` para reiniciar la meta actual y `New goal` para generar otra meta desde cero.
8. Entrar a `Agent Test`, elegir profundidad 2 y revisar que el limite del reporte sea 20.
9. Con `Help train the AI` activo (default), combinar un par con alternativas reales y elegir una salida.

## LLM Arena

El modo `Agent Test` del frontend es un **arena de solo lectura**: muestra el desafio del dia (mismo goal determinista por profundidad para todos los modelos), un podio por win rate, y las corridas recientes con el camino paso a paso de cada modelo (reproducible con playback). Las corridas se disparan **solo desde `/admin`** (seccion "LLM Arena", boton que corre los modelos en secuencia contra el goal del dia) porque cada corrida consume llamadas reales al planner; los resultados quedan publicos en `agent_runs`.

## Dashboard admin y deploy

`/admin` es un panel operativo para prender/apagar la VM GPU del modelo (estado, botones start/stop y links directos a la consola de GCP). Esta protegido con HTTP Basic auth via `apps/web/proxy.ts` usando `ADMIN_DASH_USER` / `ADMIN_DASH_PASSWORD` del entorno; si esas variables no estan definidas, `/admin` devuelve 404. El backend (`/api/admin/vm`, `src/lib/server/gcpVm.ts`) llama a la API de Compute Engine con las mismas credenciales de Google que usa Vertex (service account local o metadata server en Cloud Run). La VM objetivo se configura con `QWEN_VM_NAME` / `QWEN_VM_ZONE`.

Deploy:

* Manual: `DATABASE_URL=... ADMIN_DASH_USER=... ADMIN_DASH_PASSWORD=... ./scripts/gcp/deploy_web_cloudrun.sh` (construye la imagen y setea todas las env vars del servicio).
* CI/CD: `cloudbuild.web.yaml` en la raiz del repo reconstruye la imagen y actualiza Cloud Run **sin tocar env vars** en cada push (requiere crear el trigger de Cloud Build conectado a GitHub; el comando exacto esta comentado en el YAML).
* GPU on/off por CLI: `./scripts/gcp/model_vm.sh start|stop|status`.

## Integracion futura con SFT

La app todavia no invoca los scripts ni modelos SFT de `src/sft`. `POST /api/combine` ya conserva el contrato `CombineRequest` / `CombineResponse`, por lo que el punto de integracion esperado es reemplazar el cliente Vertex server-side por un servicio de inferencia propio cuando el modelo estudiante este listo.
