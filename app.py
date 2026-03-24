from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textSummarizer.pipeline.prediction import PredictionPipeline

# 1. Define a Data Model for the Request Body
class PredictionRequest(BaseModel):
    text: str

app = FastAPI()

# 2. Add CORS Middleware (The "Failed to Fetch" Fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows Swagger UI to communicate with the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

# 3. Improved Predict Route
@app.post("/predict")
async def predict_route(request: PredictionRequest):
    try:
        obj = PredictionPipeline()
        # Access the text via the Pydantic model: request.text
        summary = obj.predict(request.text)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)