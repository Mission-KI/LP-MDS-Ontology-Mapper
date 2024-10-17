from fastapi import FastAPI
from uvicorn import run as server_run

from edp.api import dataspace_router

app = FastAPI()

app.include_router(dataspace_router, prefix="/v1/dataspace")


def start_server():
    print("Starting server..")
    # server_run(app, host="0.0.0.0", port=8000, log_level="info", reload=False)
    # only binds to localhost
    server_run(app, port=8000, log_level="info", reload=False)
    print("Server running.")


if __name__ == "__main__":
    start_server()
