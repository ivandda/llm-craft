# Web app image for Cloud Run. Build from the repo root:
#   docker build -f apps/web/Dockerfile -t llm-craft-web apps/web
FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-slim AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:22-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
# Static assets (PWA manifest, icons, service worker) live in public/ and are
# NOT bundled into the standalone output, so copy them explicitly.
COPY --from=build /app/public ./public
ENV PORT=8080
EXPOSE 8080
CMD ["node", "server.js"]
