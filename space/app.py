"""Space entry point: ``uvicorn app:app``. All logic lives in the vendored breachlab.serve.space_app."""

from breachlab.serve.space_app import create_app

app = create_app()
