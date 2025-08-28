from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import json
from pathlib import Path


from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Post Recommendation System API",
    description="A collaborative filtering-based recommendation system",
    version="1.0.0"
)


# Mount static files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



# Global predictor instance
predictor = None

# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    title: str
    n_recommendations: Optional[int] = 5

class RecommendationResponse(BaseModel):
    query_title: str
    recommendations: List[dict]
    success: bool
    message: Optional[str] = None

# Initialize the predictor
@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        from preprocessing.predict_model import RecommendationPredictor
        predictor = RecommendationPredictor()
        
        if predictor.model.is_trained:
            logger.info("✅ Recommendation model loaded successfully")
        else:
            logger.warning("⚠️ Model not trained. Some features may not work.")
    except ImportError as e:
        logger.error(f"❌ Could not import recommendation modules: {e}")
        logger.error("Make sure all Python files are in the same directory")
        predictor = None
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictor: {str(e)}")
        predictor = None

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
        <head><title>Setup Required</title></head>
        <body>
        <h1>🚧 Setup Required</h1>
        <p>Please run the setup script first:</p>
        <pre>python setup_web.py</pre>
        </body>
        </html>
        """)

@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Recommendation system not available. Please check setup.")
    if not predictor.model.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained. Run: python main.py --full-pipeline")
    
    try:
        result = predictor.predict_single(request.title, request.n_recommendations)
        if 'error' in result:
            return RecommendationResponse(
                query_title=request.title,
                recommendations=[],
                success=False,
                message=result['error']
            )

    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/random")
async def get_random_recommendations(n_recommendations: int = 5):
    if not predictor or not predictor.model.is_trained:
        raise HTTPException(status_code=503, detail="Recommendation model not available")
    try:
        result = predictor.get_random_recommendations(n_recommendations)
        return {"query_title": result['query_title'], "recommendations": result['recommendations'], "success": True}
    except Exception as e:
        logger.error(f"Error in random recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/top-posts")
async def get_top_posts(top_n: int = 10):
    if not predictor or not predictor.model.is_trained:
        raise HTTPException(status_code=503, detail="Recommendation model not available")
    try:
        result = predictor.get_top_posts(top_n)
        return {"top_posts": result['top_posts'], "success": True}
    except Exception as e:
        logger.error(f"Error in top posts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    model_status = "available" if (predictor and predictor.model.is_trained) else "not_available"
    return {"status": "healthy", "model_status": model_status, "message": "Post Recommendation System API is running"}

@app.get("/api/similar-titles")
async def search_similar_titles(query: str, max_results: int = 10):
    if not predictor or not predictor.model.is_trained:
        return {"similar_titles": []}
    try:
        result = predictor.search_similar_titles(query, max_results)
        return result
    except Exception as e:
        logger.error(f"Error in similar titles: {str(e)}")
        return {"similar_titles": []}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Post Recommendation System Web App...")
    print("📋 Make sure you have:")
    print("   1. Trained model (run: python main.py --full-pipeline)")
    print("   2. All Python modules in the same directory")
    print("   3. Web files created (run: python setup_web.py)")
    print("🌐 Server will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
