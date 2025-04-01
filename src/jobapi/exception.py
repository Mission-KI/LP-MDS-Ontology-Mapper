from fastapi import HTTPException


class ApiClientException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(400, detail)
