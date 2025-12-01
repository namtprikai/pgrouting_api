from fastapi import FastAPI
import uvicorn
from app.routers import routes_map, search_route
from app.core.config import APP_PORT
app = FastAPI()
app.include_router(routes_map.router)
app.include_router(search_route.router)

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=APP_PORT,
        reload=True,
        workers=1,
        loop="asyncio"
    )