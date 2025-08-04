from fastapi import FastAPI
# from api.router import api_router
# from app.api.router import api_router
import api.chatbot as chatbot
import api.market_content.chatwithfiles as chatwithfiles, api.market_content.youtube_sum as youtube_sum
import api.graph_openai1 as graph_data
#import streaming.streaming as stream
from streaming.streaming import cmots_rag, web_rag,red_rag,yt_rag
from config import *
from api.fundamentals_rag.fundamental_chat2 import fund_rag
from api.fundamentals_rag.corp import corp_rag



app = FastAPI(title="Your AI-GPT Service")

app.include_router(chatbot.router)
app.include_router(chatwithfiles.router)
app.include_router(youtube_sum.router)
app.include_router(graph_data.router)
app.include_router(cmots_rag)
app.include_router(web_rag)
app.include_router(red_rag)
app.include_router(yt_rag)
app.include_router(fund_rag)
app.include_router(corp_rag)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)