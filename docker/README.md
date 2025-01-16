# Docker

## Dockerfile

In this directory you find the generic EDPS Dockerfile which is used for both the *Job API* and the *Pontus-X CLI*. The use-case is selected via entrypoint:

- Job API: `edps_jobapi` (see [Job API README](../src/jobapi/README.md))
- Pontus-X CLI: `edps_pontusx` (see [Pontus-X CLI README](../src/pontusx/README.md))

## Docker Compose for Job API

The Compose file packages EDPS with the reverse-proxy Traefik and a Postgres DB.

### Configuration

The Compose file is configured by a `.env` file in the same directory. A template can be found in `env.template`. There you can find details about the supported ENV variables.

You can configure the Docker image version (`EDPS_VERSION`), the hostname for external access (`TRAEFIK_HOSTNAME`) and the DB name and credentials (`DB_NAME`, `DB_USER` and `DB_PASSWORD`).
