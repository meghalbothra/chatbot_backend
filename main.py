import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import router as api_router  # Import the router from api.py

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Allow CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Logs a message when the application is shutting down."""
    logging.info("Shutting down the application...")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}


# Include the API router from api.py
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
