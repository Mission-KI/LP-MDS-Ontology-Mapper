# Pontus-X CLI

This is the entrypoint for an EDPS container running in the Pontus-X environment.

The CLI takes a data base directory and a DID (Dataset Identifier) to identity the correct entry.

## Usage

The CLI can be started using this command: `pontusx`

The DID is configured using ENV variable `DIDS` which should contain a JSON array of DIDs (only one is supported now!). Example: `DIDS=["83274adb"]`.

The base directory can be set with ENV variable `BASEDIR`. The default is `/data` which should be appropriate for the Pontus-X environment.

So as an entrypoint for the EDPS Pontus-X container you should use `pontusx`.

So you can test it manually in a Docker container:

`docker run -it beebucket/edps:latest pontusx`

`docker run -it -v c:/host/path:/data -e DIDS=[\"...\"] beebucket/edps:latest pontusx`
