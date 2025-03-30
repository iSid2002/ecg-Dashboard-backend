from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import ecg_router

app = FastAPI(title="ECG Dashboard API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://genuine-sprinkles-07d4ec.netlify.app",  # Your Netlify domain
        "https://ecg-dashboard-backend.onrender.com",  # Your Render backend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Include routers
app.include_router(ecg_router.router, prefix="/api")
