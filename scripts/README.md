# Scripts

## Building

To build the Docker image locally you need to run the script:

`VERSION=... ./build_docker.sh`

Locally you can e.g. use `VERSION=0.0.0`.

This first runs the script to build the wheel:

`VERSION=... ./build_wheel.sh`

## Run Docker containers

Before running Docker Compose you need to copy the file `env.template` to `.env` in directory `docker/jobapi` and adjust it if needed. For configuration see [Docker README](../docker/README.md).

Script to start all the MDS Mapper REST API containers: 
`start_jobapi.sh` 

Script to stop all the MDS MapperREST API containers: 
`stop_jobapi.sh` 
