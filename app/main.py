from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import ecg_router

app = FastAPI(title="ECG Dashboard API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ecg_router.router, prefix="/api")
