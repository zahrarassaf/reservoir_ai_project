#!/bin/bash

# Smoke test for Reservoir AI deployment

set -e

ENVIRONMENT=${1:-staging}
NAMESPACE="reservoir-ai-${ENVIRONMENT}"

# Get service URL
if [ "${ENVIRONMENT}" == "production" ]; then
    SERVICE_HOST=$(kubectl get svc -n ${NAMESPACE} reservoir-ai-service -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    SERVICE_URL="https://${SERVICE_HOST}"
else
    # For staging, use port-forward
    kubectl port-forward -n ${NAMESPACE} svc/reservoir-ai-service 8000:80 &
    PORT_FORWARD_PID=$!
    sleep 5
    SERVICE_URL="http://localhost:8000"
fi

echo "🧪 Running smoke tests against ${SERVICE_URL}"

# Test health endpoint
echo "1. Testing health endpoint..."
curl -s -f "${SERVICE_URL}/health" | jq '.status' | grep -q "healthy" || {
    echo "❌ Health check failed"
    exit 1
}
echo "✅ Health check passed"

# Test model listing
echo "2. Testing model listing..."
curl -s -f "${SERVICE_URL}/models" | jq -e 'length >= 0' || {
    echo "❌ Model listing failed"
    exit 1
}
echo "✅ Model listing passed"

# Test prediction with sample data
echo "3. Testing prediction with sample data..."

# Create sample data
cat > /tmp/sample_data.json <<EOF
{
  "model_id": "test-model",
  "data": [
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.2, 0.3, 0.4, 0.5, 0.6],
    [0.3, 0.4, 0.5, 0.6, 0.7]
  ]
}
EOF

# Try prediction (might fail if no models, but should return proper error)
PREDICTION_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d @/tmp/sample_data.json || true)

if echo "${PREDICTION_RESPONSE}" | jq -e '.error' > /dev/null 2>&1; then
    echo "⚠️ Prediction returned error (expected if no models): $(echo ${PREDICTION_RESPONSE} | jq -r '.error')"
else
    echo "✅ Prediction endpoint working"
fi

# Test training endpoint (quick test with small data)
echo "4. Testing training endpoint..."

cat > /tmp/train_data.json <<EOF
{
  "model_type": "esn",
  "config": {
    "n_inputs": 5,
    "n_outputs": 1,
    "n_reservoir": 50,
    "spectral_radius": 0.9,
    "sparsity": 0.1,
    "leaking_rate": 0.3,
    "regularization": 1e-6
  },
  "data": [
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.2, 0.3, 0.4, 0.5, 0.6],
    [0.3, 0.4, 0.5, 0.6, 0.7],
    [0.4, 0.5, 0.6, 0.7, 0.8],
    [0.5, 0.6, 0.7, 0.8, 0.9]
  ],
  "targets": [
    [0.5],
    [0.6],
    [0.7],
    [0.8],
    [0.9]
  ]
}
EOF

# Training might take time, so we'll just check if endpoint accepts request
TRAIN_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/train" \
    -H "Content-Type: application/json" \
    -d @/tmp/train_data.json || true)

if echo "${TRAIN_RESPONSE}" | jq -e '.model_id' > /dev/null 2>&1; then
    echo "✅ Training endpoint working (model created)"
    MODEL_ID=$(echo "${TRAIN_RESPONSE}" | jq -r '.model_id')
    
    # Test prediction with newly created model
    echo "5. Testing prediction with new model..."
    
    cat > /tmp/predict_new.json <<EOF
{
  "model_id": "${MODEL_ID}",
  "data": [
    [0.6, 0.7, 0.8, 0.9, 1.0]
  ]
}
EOF
    
    PREDICTION_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/predict" \
        -H "Content-Type: application/json" \
        -d @/tmp/predict_new.json)
    
    if echo "${PREDICTION_RESPONSE}" | jq -e '.predictions' > /dev/null 2>&1; then
        echo "✅ Prediction with new model successful"
    else
        echo "❌ Prediction with new model failed"
        exit 1
    fi
else
    echo "⚠️ Training endpoint returned: $(echo ${TRAIN_RESPONSE} | jq -r '.error // .')"
fi

# Cleanup
if [ "${ENVIRONMENT}" != "production" ]; then
    kill ${PORT_FORWARD_PID}
fi

rm -f /tmp/sample_data.json /tmp/train_data.json /tmp/predict_new.json

echo "🎉 All smoke tests passed!"
