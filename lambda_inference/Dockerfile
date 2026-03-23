# Container image for Lambda (XGBoost native JSON).
# Build from repo root: docker build -f lambda_inference/Dockerfile -t gunshot-lambda .
# Push to ECR; create Lambda from image; handler is set by CMD below.

FROM public.ecr.aws/lambda/python:3.12

RUN pip install --no-cache-dir "numpy>=1.26,<3" "xgboost>=2.0,<3"

COPY lambda_inference/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY models/xgboost_gunshot.json models/threshold.json /opt/ml/model/

ENV MODEL_DIR=/opt/ml/model

CMD ["lambda_function.lambda_handler"]
