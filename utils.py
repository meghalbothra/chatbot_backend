import asyncio
import uuid


async def simulate_ai_processing_time() -> None:
    """Simulates AI processing time by sleeping for 1 second."""
    await asyncio.sleep(1)


def generate_unique_session_id() -> str:
    """Generates a unique session ID."""
    return str(uuid.uuid4())


# def get_session_id(request: Request) -> str:
#     session_id = request.headers.get("session_id")
#     if not session_id:
#         raise HTTPException(status_code=400, detail="Session ID is required")
#     return session_id
