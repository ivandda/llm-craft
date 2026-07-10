# Postgres Local DB Runbook

Runbook operativo para levantar, verificar, respaldar y resetear la base local
que usa la web de llm-craft. La base corre en Docker y guarda datos fuera de git
en `var/postgres-data/`.

## Objetivo

- Servicio Docker: `postgres`
- Contenedor: `llm-craft-postgres`
- Imagen: `postgres:16-alpine`
- Base: `llm_craft`
- Usuario: `llm_craft`
- Password local: `llm_craft_dev`
- Puerto local default: `5432`
- Dataset principal: `final-10k`

URL default:

```text
postgres://llm_craft:llm_craft_dev@localhost:5432/llm_craft
```

## Prerequisitos

1. Docker Desktop abierto y con el daemon corriendo.
2. Dependencias instaladas:

```powershell
uv sync
```

3. Ejecutar comandos desde la raiz del repo:

```powershell
Set-Location D:\tpf_nlp
```

## Levantar Desde Cero

1. Iniciar Postgres:

```powershell
docker compose up -d postgres
```

2. Verificar health del contenedor:

```powershell
docker inspect --format='{{.State.Health.Status}}' llm-craft-postgres
```

Debe devolver `healthy`. Si devuelve `starting`, esperar unos segundos y repetir.

3. Aplicar migraciones:

```powershell
uv run python -m src.data.db_migrate
```

4. Importar o refrescar el dataset principal:

```powershell
uv run python -m src.data.import_final10k_to_postgres --replace-dataset final-10k
```

Este comando borra solo las filas del dataset `final-10k` y lo vuelve a importar.
No borra usuarios, sesiones, leaderboard ni datos de la app web.

## Verificacion Rapida

Estado del servicio:

```powershell
docker compose ps postgres
docker compose exec postgres pg_isready -U llm_craft -d llm_craft
```

Migraciones aplicadas:

```powershell
docker compose exec postgres psql -U llm_craft -d llm_craft -c "select version, applied_at from schema_migrations order by version;"
```

Dataset importado:

```powershell
docker compose exec postgres psql -U llm_craft -d llm_craft -c "select dataset_name, train_count, dev_count, test_count, imported_at from dataset_imports order by dataset_name;"
```

Recetas por split:

```powershell
docker compose exec postgres psql -U llm_craft -d llm_craft -c "select split, count(*) from recipe_pairs where dataset_name = 'final-10k' group by split order by split;"
```

Tablas de la app:

```powershell
docker compose exec postgres psql -U llm_craft -d llm_craft -c "select count(*) as users from users;"
docker compose exec postgres psql -U llm_craft -d llm_craft -c "select count(*) as leaderboard_entries from leaderboard_entries;"
```

Smoke test desde la web local:

```powershell
Invoke-WebRequest http://localhost:3000/api/auth/me
```

Sin sesion activa puede responder `401`. Eso es correcto. Un `500` suele indicar
que la web no puede conectarse a Postgres o que faltan migraciones.

## Uso Diario

Si `var/postgres-data/` ya existe y solo queres usar la app:

```powershell
docker compose up -d postgres
```

Si cambian migraciones o venis de otra rama:

```powershell
uv run python -m src.data.db_migrate
```

Si queres refrescar solo el dataset:

```powershell
uv run python -m src.data.import_final10k_to_postgres --replace-dataset final-10k
```

Para detener sin borrar datos:

```powershell
docker compose down
```

## Configuracion De La Web

La web usa `DATABASE_URL` si esta definida. Si no esta definida, usa la URL
default de `src/data/db.py`.

Para dejarlo explicito en `.env.local` o `.env`:

```env
DATABASE_URL=postgres://llm_craft:llm_craft_dev@localhost:5432/llm_craft
```

No commitear `.env.local`.

## Backup

Crear backup SQL:

```powershell
docker compose exec postgres pg_dump -U llm_craft -d llm_craft > backup.sql
```

Verificar que el archivo exista y tenga contenido:

```powershell
Get-Item .\backup.sql
```

Los backups pueden incluir usuarios, sesiones y datos de juego. No subirlos al
repo si contienen datos locales o datos sensibles.

## Restore

Restaurar sobre una base vacia:

```powershell
docker compose up -d postgres
Get-Content .\backup.sql | docker compose exec -T postgres psql -U llm_craft -d llm_craft
```

Si la base ya tiene datos, hacer backup primero y resetear antes de restaurar.

## Reset Completo

Esto borra toda la base local: dataset, usuarios, sesiones, leaderboard y datos
de juego. Usarlo solo si queres volver a un estado limpio.

1. Apagar Postgres:

```powershell
docker compose down
```

2. Verificar que el path a borrar sea el esperado:

```powershell
Resolve-Path .\var\postgres-data
```

Debe resolver dentro de `D:\tpf_nlp\var\postgres-data`.

3. Borrar el volumen local:

```powershell
Remove-Item -Recurse -Force .\var\postgres-data
```

4. Recrear base, migrar e importar:

```powershell
docker compose up -d postgres
uv run python -m src.data.db_migrate
uv run python -m src.data.import_final10k_to_postgres --replace-dataset final-10k
```

## Troubleshooting

### Docker no responde

Sintoma: `Cannot connect to the Docker daemon`.

Accion: abrir Docker Desktop, esperar a que termine de iniciar y repetir:

```powershell
docker compose up -d postgres
```

### Puerto 5432 ocupado

Sintoma: Docker no puede bindear `localhost:5432`.

Diagnostico:

```powershell
Get-NetTCPConnection -LocalPort 5432 -State Listen
```

Solucion temporal:

```powershell
$env:POSTGRES_PORT = "5433"
$env:DATABASE_URL = "postgres://llm_craft:llm_craft_dev@localhost:5433/llm_craft"
docker compose up -d postgres
uv run python -m src.data.db_migrate
```

Usar el mismo `DATABASE_URL` para levantar la web en esa terminal.

### La web devuelve 500 en auth o combine

Checks:

```powershell
docker compose ps postgres
docker compose exec postgres pg_isready -U llm_craft -d llm_craft
uv run python -m src.data.db_migrate
```

Si el servidor Next estaba levantado antes de iniciar Postgres, reiniciarlo.

### Falta el dataset

Sintoma: las combinaciones conocidas no aparecen o `dataset_imports` no tiene
`final-10k`.

Accion:

```powershell
uv run python -m src.data.import_final10k_to_postgres --replace-dataset final-10k
```

### Import o migraciones fallan por dependencias

Accion:

```powershell
uv sync
uv run python -m src.data.db_migrate
```

## Checklist Antes De Entregar

- `docker compose ps postgres` muestra Postgres arriba o se documenta que queda apagado.
- `schema_migrations` tiene todas las migraciones esperadas.
- `dataset_imports` contiene `final-10k`.
- `.env.local`, backups SQL y `var/postgres-data/` no estan trackeados por git.
- La web no devuelve `500` en `/api/auth/me` por falta de conexion a DB.
