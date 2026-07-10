# All-In-One VM Runbook

Runbook para crear una VM temporal en Google Cloud que hostea todo lo necesario
para probar llm-craft:

- Qwen combiner API en GPU, escuchando solo en `127.0.0.1:8000`.
- Web Next.js en `127.0.0.1:3000`.
- Nginx publico en `http://<IP_EFIMERA>/`.
- Postgres local en Docker.
- Vertex/Gemini usando el service account de la VM por metadata server.

La API del modelo no queda expuesta a internet. La web le pega por localhost.

## Arquitectura

```text
Browser -> http://<IP_EFIMERA>/ -> nginx :80
                                 -> Next :3000
                                 -> Qwen API 127.0.0.1:8000
                                 -> Postgres Docker localhost:5432
```

## Archivos Del Repo

- `scripts/gcp/create_all_in_one_vm.ps1`: corre desde Windows/local y crea la VM.
- `scripts/vm/bootstrap_all_in_one.sh`: corre dentro de Ubuntu y configura servicios.
- `docs/postgres-local-db-runbook.md`: operaciones especificas de DB local.
- `docs/qwen-combiner-vm-runbook.md`: referencia anterior solo-modelo.

## Prerequisitos Locales

1. Google Cloud CLI instalado.
2. Login activo:

```powershell
gcloud.cmd auth login
gcloud.cmd auth application-default login
```

3. Proyecto configurado:

```powershell
gcloud.cmd config set project nlp2026-498021
```

4. Estar en la raiz del repo:

```powershell
Set-Location D:\tpf_nlp
```

## Crear Y Bootstrappear La VM

Comando recomendado:

```powershell
.\scripts\gcp\create_all_in_one_vm.ps1
```

Defaults:

- Proyecto: `nlp2026-498021`
- VM: `llm-craft-all-in-one`
- Maquina: `g2-standard-4`
- GPU: `1x NVIDIA L4`
- Disco: `200GB pd-ssd`
- Rama: `dev`
- Repo: `https://github.com/ivandda/llm-craft.git`
- Web publica: puerto `80`
- Modelo: localhost `8000`, no publico
- Vertex: metadata server de GCE, sin copiar JSON por default

El script prueba varias zonas con L4. Si una zona no tiene stock, intenta la
siguiente.

Si queres usar un service account JSON especifico en vez del service account
adjunto a la VM:

```powershell
.\scripts\gcp\create_all_in_one_vm.ps1 `
  -VertexServiceAccountPath .\secrets\google_secrets.json
```

Ese archivo se copia por SSH a:

```text
~/.config/llm-craft/vertex-service-account.json
```

con permisos `600`, fuera del repo. No se commitea.

## Que Hace El Bootstrap

Dentro de la VM:

1. Instala paquetes base, Docker, Google Cloud CLI, Node.js 22, nginx y `uv`.
2. Instala driver NVIDIA si hace falta y reinicia la VM.
3. Clona o actualiza el repo en `~/llm-craft`.
4. Crea `apps/web/.env.local` con:

```env
DATABASE_URL=postgres://llm_craft:llm_craft_dev@localhost:5432/llm_craft
QWEN_COMBINER_BASE_URL=http://127.0.0.1:8000/v1
QWEN_COMBINER_API_KEY=dev-qwen-key
QWEN_COMBINER_MODEL=qwen3-4b-dpo-softce
GOOGLE_CLOUD_PROJECT=nlp2026-498021
VERTEX_LOCATION=us-central1
VERTEX_MODEL=gemini-2.5-flash
VERTEX_USE_GCE_METADATA=true
```

5. Levanta Postgres con Docker.
6. Corre migraciones y reimporta `final-10k`.
7. Descarga el adapter desde GCS:

```text
gs://llm-craft-bucket/runs/2026-07-09_0641_qwen3_4b_dpo_softce/best_adapter
```

8. Verifica SHA del adapter:

```text
f3a608751c65d6ef479be3fb4edd36c9f7eac5737f0e01d7be43e866dcdcaff8
```

9. Instala y arranca servicios `systemd`:

- `qwen-combiner`
- `llm-craft-web`
- `nginx`

## Reboot Por Driver

La primera ejecucion puede reiniciar la VM para activar el driver NVIDIA. El
script local espera SSH y reintenta una vez automaticamente.

Si se corta a mano, volver a ejecutar:

```powershell
.\scripts\gcp\create_all_in_one_vm.ps1
```

El bootstrap es idempotente: vuelve a usar la misma VM si ya existe.

## Obtener IP Y Abrir La Web

Al final el script imprime:

```text
Web: http://<IP_EFIMERA>/
```

Tambien se puede obtener manualmente:

```powershell
$zone = gcloud.cmd compute instances list `
  --project nlp2026-498021 `
  --filter "name=(llm-craft-all-in-one)" `
  --format "value(zone.basename())"

$ip = gcloud.cmd compute instances describe llm-craft-all-in-one `
  --project nlp2026-498021 `
  --zone $zone `
  --format "value(networkInterfaces[0].accessConfigs[0].natIP)"

$ip
```

Abrir:

```text
http://<IP_EFIMERA>/
```

## Uso Esperado En La Web

Para probar el modelo propio:

1. Entrar a la web.
2. Loguearse o crear usuario.
3. Elegir `Qwen DPO SoftCE` en el selector de combinador.
4. Combinar elementos.

Gemini sigue siendo default en la UI y usa Vertex. Por default la web toma un
token del metadata server de GCE con el service account adjunto a la VM. Qwen
queda listo en la misma VM y se usa al seleccionar `Qwen DPO SoftCE`.

El service account de la VM necesita permisos para Vertex AI, por ejemplo
`roles/aiplatform.user` sobre el proyecto. Si la VM usa el default Compute
Engine service account y ese service account no tiene permisos, Vertex va a
devolver `403`.

## Validacion Operativa

SSH:

```powershell
gcloud.cmd compute ssh llm-craft-all-in-one --zone <ZONE> --project nlp2026-498021
```

En la VM:

```bash
nvidia-smi
curl -s http://127.0.0.1:8000/health
curl -I http://127.0.0.1/
sudo systemctl status qwen-combiner --no-pager
sudo systemctl status llm-craft-web --no-pager
sudo systemctl status nginx --no-pager
docker compose -f ~/llm-craft/compose.yml ps postgres
grep -E "GOOGLE_CLOUD_PROJECT|VERTEX_LOCATION|VERTEX_USE_GCE_METADATA|GOOGLE_APPLICATION_CREDENTIALS" ~/llm-craft/apps/web/.env.local
```

Test de inferencia desde la VM:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-qwen-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b-dpo-softce",
    "messages": [
      {"role": "system", "content": "You combine two concepts into one resulting concept."},
      {"role": "user", "content": "Given two concepts, combine them into one resulting concept.\n\nConcept A: fire\nConcept B: water\n\nReturn only the resulting concept."}
    ],
    "max_tokens": 16,
    "temperature": 0
  }' | jq
```

Resultado esperado: `steam`.

Test de token Vertex por metadata server:

```bash
curl -s \
  -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" |
  jq '{has_access_token: has("access_token"), expires_in}'
```

Si usaste `-VertexServiceAccountPath`, validar que el archivo existe y no esta
dentro del repo:

```bash
ls -l ~/.config/llm-craft/vertex-service-account.json
grep GOOGLE_APPLICATION_CREDENTIALS ~/llm-craft/apps/web/.env.local
```

## Logs

```bash
sudo journalctl -u qwen-combiner -f
sudo journalctl -u llm-craft-web -f
sudo journalctl -u nginx -f
docker logs -f llm-craft-postgres
```

## Deploy De Cambios Nuevos

Despues de pushear cambios a `dev`, correr otra vez el bootstrap desde local:

```powershell
.\scripts\gcp\create_all_in_one_vm.ps1
```

Eso hace `git pull`, reescribe `.env.local`, aplica migraciones, reimporta
`final-10k`, rebuilda Next y reinicia servicios.

Para cambiar proyecto/region/modelo de Vertex:

```powershell
.\scripts\gcp\create_all_in_one_vm.ps1 `
  -VertexProject nlp2026-498021 `
  -VertexLocation us-central1 `
  -VertexModel gemini-2.5-flash
```

Si no queres tocar infraestructura y solo correr el bootstrap:

```powershell
.\scripts\gcp\create_all_in_one_vm.ps1 -SkipBootstrap

gcloud.cmd compute scp .\scripts\vm\bootstrap_all_in_one.sh `
  llm-craft-all-in-one:~/bootstrap_all_in_one.sh `
  --zone <ZONE> `
  --project nlp2026-498021

gcloud.cmd compute ssh llm-craft-all-in-one `
  --zone <ZONE> `
  --project nlp2026-498021 `
  --command "chmod +x ~/bootstrap_all_in_one.sh && bash ~/bootstrap_all_in_one.sh"
```

## Firewall

Reglas creadas:

- `allow-llm-craft-all-in-one-http`: abre `tcp:80` a internet.
- `allow-llm-craft-all-in-one-ssh`: abre `tcp:22` solo a la IP publica local actual.

No se abre `tcp:8000`.

Si cambia tu IP y no podes entrar por SSH:

```powershell
$publicIp = (Invoke-RestMethod -Uri "https://ifconfig.me/ip").Trim()

gcloud.cmd compute firewall-rules update allow-llm-craft-all-in-one-ssh `
  --project nlp2026-498021 `
  --source-ranges "$publicIp/32"
```

## Apagar, Prender Y Borrar

Apagar para no pagar computo:

```powershell
gcloud.cmd compute instances stop llm-craft-all-in-one --zone <ZONE> --project nlp2026-498021
```

Prender:

```powershell
gcloud.cmd compute instances start llm-craft-all-in-one --zone <ZONE> --project nlp2026-498021
```

Despues de prender, la IP efimera puede cambiar. Volver a obtenerla. Los
servicios `qwen-combiner`, `llm-craft-web`, `nginx` y el contenedor Postgres
deberian arrancar solos.

Borrar al terminar:

```powershell
gcloud.cmd compute instances delete llm-craft-all-in-one --zone <ZONE> --project nlp2026-498021
gcloud.cmd compute firewall-rules delete allow-llm-craft-all-in-one-http --project nlp2026-498021
gcloud.cmd compute firewall-rules delete allow-llm-craft-all-in-one-ssh --project nlp2026-498021
```

## Troubleshooting

### La web abre pero Qwen falla

En la VM:

```bash
sudo systemctl status qwen-combiner --no-pager
sudo journalctl -u qwen-combiner -n 120 --no-pager
curl -s http://127.0.0.1:8000/health
```

Si el modelo esta cargando, esperar. La primera carga puede tardar varios
minutos.

### La web devuelve errores de DB

En la VM:

```bash
cd ~/llm-craft
docker compose ps postgres
uv run python -m src.data.db_migrate
uv run python -m src.data.import_final10k_to_postgres --replace-dataset final-10k
sudo systemctl restart llm-craft-web
```

### Vertex devuelve 403

El service account adjunto a la VM no tiene permisos suficientes para Vertex AI.
Ver el service account:

```powershell
gcloud.cmd compute instances describe llm-craft-all-in-one `
  --zone <ZONE> `
  --project nlp2026-498021 `
  --format "value(serviceAccounts[0].email)"
```

Dar permisos en el proyecto:

```powershell
gcloud.cmd projects add-iam-policy-binding nlp2026-498021 `
  --member "serviceAccount:<SERVICE_ACCOUNT_EMAIL>" `
  --role "roles/aiplatform.user"
```

Despues reiniciar la web:

```bash
sudo systemctl restart llm-craft-web
```

### Vertex no consigue token

Validar metadata server:

```bash
curl -i \
  -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
```

Si preferis evitar metadata, recrear/bootstrappear pasando un JSON:

```powershell
.\scripts\gcp\create_all_in_one_vm.ps1 `
  -VertexServiceAccountPath .\secrets\google_secrets.json
```

### Nginx responde 502

```bash
sudo systemctl status llm-craft-web --no-pager
sudo journalctl -u llm-craft-web -n 120 --no-pager
curl -I http://127.0.0.1:3000/
```

### No hay stock de GPU

Editar `-Zones` al ejecutar el script o pasar zonas manualmente:

```powershell
.\scripts\gcp\create_all_in_one_vm.ps1 -Zones @("us-west1-a","us-east1-b")
```

### Repo privado o credenciales Git

El script clona por HTTPS. Si el repo no es publico desde la VM, usar una URL
con credenciales temporales o subir el codigo manualmente. No commitear tokens.

## Notas

- La VM es temporal y usa IP efimera.
- No subir adapters, pesos, `.env.local`, backups SQL ni `var/postgres-data/`.
- No subir service account JSON. Si se usa, queda fuera del repo en la VM.
- Para demo rapida conviene esta VM unica. Para produccion, separar web, DB y GPU.
