# Frontend Next.js

La aplicacion jugable vive en `apps/web` y esta implementada como una app Next.js. Por ahora usa contratos y endpoints mock para validar la experiencia de juego antes de conectar modelos entrenados.

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

## Credenciales y modos

El backend mock mantiene usuarios en memoria y trae un usuario seeded:

```text
usuario: admin
password: admin
```

La UI incluye dos modos:

* `Sandbox`: inventario inicial amplio para explorar combinaciones.
* `Goal`: inventario inicial acotado y objetivo fijo para completar una meta.

Los cambios de perfil, logros destacados y leaderboard se guardan contra endpoints mock en memoria. Se pierden al reiniciar el proceso de Next.js.

## Endpoints mock

Los endpoints viven bajo `apps/web/app/api`:

* `POST /api/auth/login`
* `POST /api/auth/register`
* `POST /api/auth/logout`
* `GET /api/auth/me`
* `POST /api/combine`
* `GET /api/leaderboard`
* `POST /api/leaderboard`
* `PATCH /api/profile`

Los tipos compartidos estan en `apps/web/src/lib/types.ts`. La combinacion de elementos usa recetas seed y un fallback deterministico en `apps/web/src/lib/craft.ts`.

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
5. Probar una combinacion no conocida para validar el fallback mock.
6. Cambiar entre `Sandbox` y `Goal` y revisar el leaderboard mock.

## Integracion futura con SFT

La app todavia no invoca los scripts ni modelos SFT de `src/sft`. El punto de integracion esperado es reemplazar o extender `POST /api/combine` para llamar a un servicio de inferencia y conservar el contrato `CombineRequest` / `CombineResponse`.
