# Docker

## Dockerfile

In this directory you find the Dockerfile which is used for generating a docker image for the mds mapper.

## Docker Compose for Job API

The Compose file packages MDS Mapper with the reverse-proxy Traefik and a Postgres DB.

### Configuration

The Compose file is configured by a `.env` file in the same directory. A template can be found in `env.template`. There you can find details about the supported ENV variables.

You can configure the Docker image version (`MDS_MAPPER_VERSION`), the hostname for external access (`TRAEFIK_HOSTNAME`) and the DB name and credentials (`DB_NAME`, `DB_USER` and `DB_PASSWORD`).
