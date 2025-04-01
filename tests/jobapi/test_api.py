from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from jobapi.__main__ import init_fastapi
from jobapi.config import AppConfig
from jobapi.types import JobData, JobState, JobView


@pytest.fixture
def app(path_work):
    return init_fastapi(AppConfig(working_dir=path_work))


@pytest.fixture
def test_client(app):
    return TestClient(app)


@pytest.mark.slow
async def test_api(test_client, user_provided_data, asset_path: Path):
    job_data = JobData(user_provided_edp_data=user_provided_data)
    response = test_client.post("/v1/dataspace/analysisjob", content=job_data.model_dump_json(by_alias=True))
    assert response.status_code == 200, response.text
    job = JobView.model_validate(response.json())
    assert job.state == JobState.WAITING_FOR_DATA

    files = {"upload_file": (asset_path.name, asset_path.read_bytes())}
    response = test_client.post(f"/v1/dataspace/analysisjob/{job.job_id}/data", files=files)
    job = extract_job_view(response)
    assert job.state in [JobState.QUEUED, JobState.PROCESSING]

    # Wait until the processing is completed
    while job.state in [JobState.QUEUED, JobState.PROCESSING]:
        job = get_status(test_client, job.job_id)
    assert job.state == JobState.COMPLETED

    # Check for valid result
    response = test_client.get(f"/v1/dataspace/analysisjob/{job.job_id}/result")
    assert response.status_code == 200, response.text
    # Check for report
    response = test_client.get(f"/v1/dataspace/analysisjob/{job.job_id}/report")
    assert response.status_code == 200, response.text
    # Check for log
    response = test_client.get(f"/v1/dataspace/analysisjob/{job.job_id}/log")
    assert response.status_code == 200, response.text


async def test_api_client_error(test_client):
    random_uuid = uuid4()
    response = test_client.get(f"/v1/dataspace/analysisjob/{random_uuid}/result")
    assert response.status_code == 400
    assert response.json()["detail"] is not None


def get_status(client: TestClient, job_id: UUID):
    response: Response = client.get(f"/v1/dataspace/analysisjob/{job_id}/status")
    return extract_job_view(response)


def extract_job_view(response: Response) -> JobView:
    assert response.status_code == 200, response.text
    return JobView.model_validate(response.json())
