import json
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes_artifacts import router as artifacts_router
from app.api.routes_compile import router as compile_router
from app.api.routes_health import router as health_router
from app.api.routes_packets import router as packets_router
from app.api.routes_projects import router as projects_router
from app.storage.db import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    print(json.dumps({"event": "app_startup", "service": "purtera_evidence_compiler"}), file=sys.stderr)
    yield


app = FastAPI(title="Purtera Evidence Compiler MVP", lifespan=lifespan)


app.include_router(health_router)
app.include_router(projects_router)
app.include_router(artifacts_router)
app.include_router(compile_router)
app.include_router(packets_router)
