import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.query_handler import QueryHandler

# Create FastAPI instance
app = FastAPI()

# Setup CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request body model
class SearchRequest(BaseModel):
    query: str
    data_type: str
    num_search_results: int


@app.post("/api/query")
async def search_completion(request: SearchRequest):
    try:
        sh = QueryHandler()
        result = await asyncio.to_thread(sh.query, query=request.query)
        return jsonable_encoder(result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later.",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=6000, reload=True)
