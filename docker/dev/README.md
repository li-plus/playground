# Docker Dev

Start a docker container for development.
```sh
docker compose up -d
docker compose exec -u $(id -u):$(id -g) dev bash
```
