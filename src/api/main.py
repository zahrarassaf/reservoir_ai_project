"""
FastAPI application for Reservoir AI.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np
import joblib
import tempfile
from pathlib import Path
import logging
from datetime import datetime
import uuid

from ..models.esn import EchoStateNetwork, ESNConfig
from ..models.deep_esn import DeepEchoStateNetwork, DeepESNConfig
from ..data.preprocessor import SPE9Preprocessor, PreprocessingConfig
from ..utils.metrics import PetroleumMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Reservoir AI API",
    description="Industrial-grade Reservoir Computing for petroleum reservoir simulation",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# In-memory cache for loaded models
model_cache = {}


# Pydantic models for request/response
class TrainRequest(BaseModel):
    """Request model for training."""
    
    model_type: str = Field(default="esn", description="Type of model: 'esn' or 'deep_esn'")
    config: Dict[str, Any] = Field(description="Model configuration")
    data: List[List[float]] = Field(description="Input data as 2D array")
    targets: List[List[float]] = Field(description="Target data as 2D array")
    
    @validator('data', 'targets')
    def validate_data_shape(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        if not all(len(row) == len(v[0]) for row in v):
            raise ValueError("All rows must have the same length")
        return v
    
    @validator('data')
    def validate_data_dimensions(cls, v, values):
        if 'config' in values:
            config = values['config']
            if 'n_inputs' in config:
                expected_cols = config['n_inputs']
                actual_cols = len(v[0])
                if actual_cols != expected_cols:
                    raise ValueError(f"Expected {expected_cols} input columns, got {actual_cols}")
        return v


class PredictRequest(BaseModel):
    """Request model for prediction."""
    
    model_id: str = Field(description="ID of the trained model")
    data: List[List[float]] = Field(description="Input data for prediction")
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        return v


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_id: str
    model_type: str
    config: Dict[str, Any]
    training_stats: Optional[Dict[str, Any]]
    created_at: datetime
    input_shape: Optional[Tuple[int, int]]
    output_shape: Optional[Tuple[int, int]]


class PredictionResponse(BaseModel):
    """Prediction response."""
    
    predictions: List[List[float]]
    model_id: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    timestamp: datetime
    models_loaded: int
    uptime_seconds: float


# Dependency for model loading
def get_model(model_id: str):
    """Load model from cache or disk."""
    if model_id in model_cache:
        return model_cache[model_id]
    
    model_path = MODELS_DIR / f"{model_id}.pkl"
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} not found"
        )
    
    try:
        model = joblib.load(model_path)
        model_cache[model_id] = model
        return model
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Reservoir AI API")
    
    # Load any pre-trained models
    for model_file in MODELS_DIR.glob("*.pkl"):
        try:
            model_id = model_file.stem
            model = joblib.load(model_file)
            model_cache[model_id] = model
            logger.info(f"Loaded model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to load model {model_file}: {e}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Reservoir AI API")
    model_cache.clear()


# Routes
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {"message": "Reservoir AI API", "version": "0.1.0"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import time
    from . import __version__
    
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now(),
        models_loaded=len(model_cache),
        uptime_seconds=time.time() - app.start_time if hasattr(app, 'start_time') else 0
    )


@app.post("/train", status_code=status.HTTP_201_CREATED)
async def train_model(request: TrainRequest):
    """Train a new model."""
    try:
        logger.info(f"Training {request.model_type} model")
        
        # Convert data to numpy arrays
        X = np.array(request.data, dtype=np.float32)
        y = np.array(request.targets, dtype=np.float32)
        
        # Create model configuration
        if request.model_type == "esn":
            config = ESNConfig(**request.config)
            model = EchoStateNetwork(config)
        elif request.model_type == "deep_esn":
            config = DeepESNConfig(**request.config)
            model = DeepEchoStateNetwork(config)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model type: {request.model_type}"
            )
        
        # Train model
        training_stats = model.fit(X, y)
        
        # Generate unique model ID
        model_id = str(uuid.uuid4())
        
        # Save model
        model_path = MODELS_DIR / f"{model_id}.pkl"
        joblib.dump(model, model_path)
        
        # Add to cache
        model_cache[model_id] = model
        
        # Prepare response
        model_info = ModelInfo(
            model_id=model_id,
            model_type=request.model_type,
            config=request.config,
            training_stats=training_stats,
            created_at=datetime.now(),
            input_shape=X.shape,
            output_shape=y.shape,
        )
        
        logger.info(f"Model trained successfully: {model_id}")
        
        return model_info
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest):
    """Make predictions with a trained model."""
    import time
    
    start_time = time.time()
    
    try:
        # Load model
        model = get_model(request.model_id)
        
        # Convert data
        X = np.array(request.data, dtype=np.float32)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Convert to list
        predictions_list = predictions.tolist()
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return PredictionResponse(
            predictions=predictions_list,
            model_id=request.model_id,
            inference_time_ms=inference_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models."""
    models_info = []
    
    for model_id, model in model_cache.items():
        try:
            model_info = ModelInfo(
                model_id=model_id,
                model_type=type(model).__name__,
                config=getattr(model, 'config', {}).__dict__,
                training_stats=getattr(model, 'training_stats', {}),
                created_at=datetime.fromtimestamp(
                    (MODELS_DIR / f"{model_id}.pkl").stat().st_mtime
                ),
                input_shape=None,
                output_shape=None,
            )
            models_info.append(model_info)
        except Exception as e:
            logger.warning(f"Failed to get info for model {model_id}: {e}")
    
    return models_info


@app.get("/models/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """Get information about a specific model."""
    model = get_model(model_id)
    
    model_info = ModelInfo(
        model_id=model_id,
        model_type=type(model).__name__,
        config=getattr(model, 'config', {}).__dict__,
        training_stats=getattr(model, 'training_stats', {}),
        created_at=datetime.fromtimestamp(
            (MODELS_DIR / f"{model_id}.pkl").stat().st_mtime
        ),
        input_shape=None,
        output_shape=None,
    )
    
    return model_info


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model."""
    try:
        # Remove from cache
        if model_id in model_cache:
            del model_cache[model_id]
        
        # Delete file
        model_path = MODELS_DIR / f"{model_id}.pkl"
        if model_path.exists():
            model_path.unlink()
        
        return {"message": f"Model {model_id} deleted successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {str(e)}"
        )


@app.post("/evaluate")
async def evaluate_model(
    model_id: str = Query(..., description="Model ID"),
    data: List[List[float]] = Query(..., description="Input data"),
    targets: List[List[float]] = Query(..., description="True targets")
):
    """Evaluate model performance."""
    try:
        # Load model
        model = get_model(model_id)
        
        # Convert data
        X = np.array(data, dtype=np.float32)
        y_true = np.array(targets, dtype=np.float32)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = PetroleumMetrics.comprehensive_metrics(y_true, y_pred)
        
        # Format report
        report = PetroleumMetrics.format_metrics_report(metrics)
        
        return {
            "model_id": model_id,
            "metrics": metrics,
            "report": report,
            "predictions": y_pred.tolist()
        }
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@app.post("/optimize")
async def optimize_hyperparameters(
    model_type: str = Query("esn", description="Model type"),
    data: List[List[float]] = Query(..., description="Input data"),
    targets: List[List[float]] = Query(..., description="Target data"),
    optimization_config: Dict[str, Any] = Query(..., description="Optimization configuration")
):
    """Optimize hyperparameters using Bayesian optimization."""
    try:
        from ..optimization.bayesian_optimizer import (
            ESNBayesianOptimizer, OptimizationConfig
        )
        
        # Convert data
        X = np.array(data, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        
        # Create optimizer
        opt_config = OptimizationConfig(**optimization_config)
        optimizer = ESNBayesianOptimizer(X, y, opt_config)
        
        # Run optimization
        results = optimizer.optimize()
        
        # Train final model with best parameters
        best_model = optimizer.create_best_model(X, y)
        
        # Save model
        model_id = str(uuid.uuid4())
        model_path = MODELS_DIR / f"{model_id}.pkl"
        joblib.dump(best_model, model_path)
        model_cache[model_id] = best_model
        
        return {
            "model_id": model_id,
            "optimization_results": results,
            "best_parameters": results['best_params'],
            "best_score": results['best_score']
        }
    
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "Contact administrator"
        }
    )
