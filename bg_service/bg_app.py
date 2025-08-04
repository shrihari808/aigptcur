from fastapi import FastAPI
import api.investor_stories as investor_stories, api.news_embeddings as news_embeddings,api.trending_stocks as trending_stocks
#import api.fundamentals_embeddings as fundamentals_embeddings
# import api.suggestedprompts as suggestedprompts
import api.autocomplete_news_questions as autocomplete_news_questions
import signal
import asyncio
import uvicorn
import sys
from config import *

app = FastAPI()

app.include_router(investor_stories.router)
app.include_router(news_embeddings.router)
# app.include_router(reddit_embeddings.router)
app.include_router(trending_stocks.router)
#app.include_router(fundamentals_embeddings.router)
# app.include_router(suggestedprompts.router)
app.include_router(autocomplete_news_questions.router)



async def shutdown_event():
    print("Performing cleanup tasks...")
    # Add any cleanup tasks here if needed
    await asyncio.sleep(1)  # Give some time for tasks to complete
    print("Shutdown complete")
    sys.exit(0)

def shutdown_handler(signum, frame):
    print("Shutting down gracefully...")
    loop = asyncio.get_event_loop()
    loop.create_task(shutdown_event())
    loop.stop()

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

@app.router.lifespan_context
async def lifespan(app: FastAPI):
    # Startup code here (if needed)
    print("Server is starting up...")
    yield
    # Shutdown code here
    await shutdown_event()

# if __name__ == "__main__":
#     try:
#         uvicorn.run(app, host="0.0.0.0", port=8000)
#     except asyncio.CancelledError:
#         print("Event loop was cancelled")