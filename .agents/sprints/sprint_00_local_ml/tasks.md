# Sprint 0: Local ML Pipeline Setup

**Status**: ✅ Complete  
**Completion Date**: 2026-01-30  
**Sprint Goal**: Build complete local ML pipeline with data generation, training, evaluation, and inference API

---

## Summary

Established a complete local machine learning pipeline using scikit-learn. The pipeline generates synthetic data, trains a RandomForest classifier, evaluates it, and serves predictions via FastAPI.

**Results**:
- Training accuracy: 99.25%
- Test accuracy: 90.0%
- All unit tests passing
- FastAPI server operational

---

## Completed Tasks

### [HUMAN] 0.1: Install Required Tools on MacBook ✅
**Status**: ✅ Complete  
**Verification**: `python3 --version` shows 3.10+

### [HUMAN] 0.2: Install Docker Desktop ✅
**Status**: ✅ Complete  
**Verification**: `docker run hello-world` works

### [HUMAN] 0.3: Create Project Virtual Environment ✅
**Status**: ✅ Complete  
**Verification**: `.venv/` directory created and activated

### [HUMAN] 0.4: Install DVC and MLflow ✅
**Status**: ✅ Complete  
**Verification**: `dvc version` and `mlflow --version` work

### [AI] 0.1: Create Project Directory Structure ✅
**Status**: ✅ Complete  
**Delivered**: Full directory structure for ML project

**Directories Created**:
```
projects/synth_tabular_classification/
├── src/
│   ├── data/
│   ├── model/
│   └── inference/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── metrics/
├── configs/
├── tests/
└── notebooks/
```

### [AI] 0.2: Create requirements.txt ✅
**Status**: ✅ Complete  
**Delivered**: Complete Python dependencies

**Key Dependencies**:
- scikit-learn>=1.3.0
- pandas>=2.0.0
- mlflow>=2.10.0
- dvc[gs]>=3.30.0
- fastapi>=0.109.0
- pytest>=7.4.0
- ruff>=0.1.0

### [AI] 0.3: Create Configuration File ✅
**Status**: ✅ Complete  
**Delivered**: `configs/config.yaml` with data and model parameters

### [AI] 0.4: Create Data Generation Script ✅
**Status**: ✅ Complete  
**Delivered**: `src/data/generate.py` using sklearn.datasets.make_classification

### [AI] 0.5: Create Data Preprocessing Script ✅
**Status**: ✅ Complete  
**Delivered**: `src/data/preprocess.py` with train/test split

### [AI] 0.6: Create Model Training Script ✅
**Status**: ✅ Complete  
**Delivered**: `src/model/train.py` with MLflow tracking

**Features**:
- RandomForest classifier
- MLflow experiment tracking
- Parameters and metrics logged
- Model saved locally

### [AI] 0.7: Create Model Evaluation Script ✅
**Status**: ✅ Complete  
**Delivered**: `src/model/evaluate.py` with comprehensive metrics

**Metrics Tracked**:
- Accuracy, F1, Precision, Recall
- Classification report
- Confusion matrix

### [AI] 0.8: Create Prediction Script ✅
**Status**: ✅ Complete  
**Delivered**: `src/inference/predict.py` for batch predictions

### [AI] 0.9: Create FastAPI Inference Server ✅
**Status**: ✅ Complete  
**Delivered**: `src/inference/server.py` with REST API

**Endpoints**:
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /info` - Model information
- `POST /predict` - Make predictions

### [AI] 0.10: Create Unit Tests ✅
**Status**: ✅ Complete  
**Delivered**: Test suite in `tests/`

**Test Files**:
- `test_data.py` - Data generation tests
- `test_model.py` - Model training tests
- `test_inference.py` - API endpoint tests

### [AI] 0.11: Create .gitignore for Project ✅
**Status**: ✅ Complete  
**Delivered**: Comprehensive .gitignore for Python/ML project

---

## Files Created

- `projects/synth_tabular_classification/requirements.txt`
- `projects/synth_tabular_classification/configs/config.yaml`
- `projects/synth_tabular_classification/src/data/generate.py`
- `projects/synth_tabular_classification/src/data/preprocess.py`
- `projects/synth_tabular_classification/src/model/train.py`
- `projects/synth_tabular_classification/src/model/evaluate.py`
- `projects/synth_tabular_classification/src/inference/predict.py`
- `projects/synth_tabular_classification/src/inference/server.py`
- `projects/synth_tabular_classification/tests/test_data.py`
- `projects/synth_tabular_classification/tests/test_model.py`
- `projects/synth_tabular_classification/tests/test_inference.py`
- `projects/synth_tabular_classification/.gitignore`

---

## Verification Commands

```bash
cd projects/synth_tabular_classification

# Install dependencies
pip install -r requirements.txt

# Run pipeline manually
python -m src.data.generate
python -m src.data.preprocess
python -m src.model.train
python -m src.model.evaluate
python -m src.inference.predict

# Run tests
pytest tests/ -v

# Start API server
uvicorn src.inference.server:app --host 0.0.0.0 --port 8000

# Test API
curl http://localhost:8000/health
curl http://localhost:8000/info
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0]]}'
```

---

## Lessons Learned

1. **MLflow Local Setup**: File-based backend is sufficient for local development
2. **FastAPI Patterns**: Use lifespan context managers for startup/shutdown
3. **Test Structure**: Separate tests by component (data, model, inference)
4. **Configuration**: YAML config keeps parameters organized and version-controlled

---

## Next Sprint

**Sprint 1**: DVC Pipeline Setup
- Create DVC pipeline file
- Create params.yaml for DVC
- Initialize DVC in project
