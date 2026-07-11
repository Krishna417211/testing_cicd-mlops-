# Taxi Fare Project

## Folder structure
```
taxi-project/
├── data/
│   └── sample.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── app.py
├── models/
│   └── taxi_model.keras   (created after training)
├── tests/
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_app.py
├── .github/workflows/ci-cd.yml
├── Dockerfile
├── requirements.txt
├── .pylintrc
└── pytest.ini
```

## Run locally
```bash
pip install -r requirements.txt

python src/preprocess.py     # 1. clean data  -> data/taxi_clean.csv
python src/train.py          # 2. train       -> models/taxi_model.keras
pytest                       # 3. run tests
pylint src/                  # 4. lint
uvicorn src.app:app --reload # 5. serve API -> http://localhost:8000/docs
```

Test the API:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"passenger_count":1,"trip_distance":2.5,"fare_amount":10,"tip_amount":2}'
```

## Deployment (Docker)
```bash
docker build -t taxi-fare-api .
docker run -p 8000:8000 taxi-fare-api
```

## CI/CD Pipeline
On every push to `main`, `.github/workflows/ci-cd.yml` runs three stages:
1. **test**  – pylint + pytest
2. **train** – preprocess + train, uploads the model as an artifact
3. **deploy**– builds the Docker image and pushes it to Docker Hub

Add these GitHub secrets (Settings → Secrets and variables → Actions):
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
```
