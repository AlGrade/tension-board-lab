# syntax=docker/dockerfile:1.7

# Browser build. npm and every build dependency stop at this stage.
FROM node:24.19.0-alpine AS build

WORKDIR /build

COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund

COPY index.html tsconfig.json vite.config.ts ./
COPY public/ ./public/
COPY src/ ./src/

RUN npm run build


# Static gateway, and the only service in the deployment. Everything this app does — grading,
# generating, rendering — happens in the browser, so hosting it is serving files. Port 8080 lets
# nginx run without root or NET_BIND_SERVICE.
FROM nginx:1.29-alpine AS static

COPY nginx-static.conf /etc/nginx/nginx.conf
COPY --from=build --chown=nginx:nginx /build/dist/ /usr/share/nginx/html/

USER nginx:nginx

EXPOSE 8080
STOPSIGNAL SIGQUIT

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1

# Skip the parent image's root-oriented config mutation scripts. The config and assets are final.
ENTRYPOINT ["/usr/sbin/nginx"]
CMD ["-g", "daemon off;"]
