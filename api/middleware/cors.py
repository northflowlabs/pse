"""
CORS configuration for the PSE API.
"""
from starlette.middleware.cors import CORSMiddleware

ALLOWED_ORIGINS = [
    "http://localhost:3000",   # local React dev server
    "http://localhost:5173",   # Vite dev server
    "https://northflow.no",
    "https://app.northflow.no",
]

CORS_KWARGS = dict(
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
