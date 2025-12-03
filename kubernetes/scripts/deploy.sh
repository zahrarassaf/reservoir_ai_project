#!/bin/bash

# Reservoir AI Deployment Script
# Usage: ./deploy.sh [environment]

set -e

ENVIRONMENT=${1:-staging}
VERSION=${2:-$(git rev-parse --short HEAD)}
REGISTRY="yourregistry.com/reservoir-ai"
NAMESPACE="reservoir-ai-${ENVIRONMENT}"

echo "🚀 Deploying Reservoir AI to ${ENVIRONMENT}"
echo "📦 Version: ${VERSION}"
echo "📝 Registry: ${REGISTRY}"
echo "📁 Namespace: ${NAMESPACE}"

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { echo "kubectl required but not installed. Aborting." >&2; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "helm required but not installed. Aborting." >&2; exit 1; }

# Create namespace if it doesn't exist
if ! kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
    echo "📝 Creating namespace ${NAMESPACE}"
    kubectl create namespace "${NAMESPACE}"
fi

# Set context
kubectl config set-context --current --namespace="${NAMESPACE}"

# Create secrets if not exists
if [ ! -f "secrets/${ENVIRONMENT}.env" ]; then
    echo "⚠️ Warning: secrets file not found for ${ENVIRONMENT}"
    echo "Creating default secrets..."
    
    # Create secrets from template
    envsubst < kubernetes/secrets-template.yaml > kubernetes/secrets-${ENVIRONMENT}.yaml
else
    echo "🔐 Loading secrets from secrets/${ENVIRONMENT}.env"
    source "secrets/${ENVIRONMENT}.env"
    
    # Create secrets
    cat > kubernetes/secrets-${ENVIRONMENT}.yaml <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: reservoir-ai-secrets
  namespace: ${NAMESPACE}
type: Opaque
data:
  MLFLOW_USERNAME: $(echo -n "${MLFLOW_USERNAME}" | base64)
  MLFLOW_PASSWORD: $(echo -n "${MLFLOW_PASSWORD}" | base64)
  MINIO_ACCESS_KEY: $(echo -n "${MINIO_ACCESS_KEY}" | base64)
  MINIO_SECRET_KEY: $(echo -n "${MINIO_SECRET_KEY}" | base64)
  DATABASE_URL: $(echo -n "${DATABASE_URL}" | base64)
EOF
fi

# Apply secrets
echo "🔐 Applying secrets..."
kubectl apply -f "kubernetes/secrets-${ENVIRONMENT}.yaml"

# Build and push Docker image
echo "🐳 Building Docker image..."
docker build -t "${REGISTRY}:${VERSION}" -t "${REGISTRY}:latest" .

echo "📤 Pushing Docker image..."
docker push "${REGISTRY}:${VERSION}"
docker push "${REGISTRY}:latest"

# Update deployment with new image
echo "🎯 Updating deployment..."
cat kubernetes/deployment.yaml | \
    sed "s|yourregistry/reservoir-ai:latest|${REGISTRY}:${VERSION}|g" | \
    kubectl apply -f -

# Wait for rollout
echo "⏳ Waiting for deployment rollout..."
kubectl rollout status deployment/reservoir-ai-api --timeout=300s

# Run smoke tests
echo "🧪 Running smoke tests..."
if [ -f "scripts/smoke-test.sh" ]; then
    ./scripts/smoke-test.sh "${ENVIRONMENT}"
fi

# Check deployment health
echo "🏥 Checking deployment health..."
kubectl get pods -l app=reservoir-ai

# Get service URL
if [ "${ENVIRONMENT}" == "production" ]; then
    SERVICE_URL=$(kubectl get svc reservoir-ai-service -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    echo "🌐 Service URL: https://${SERVICE_URL}"
fi

echo "✅ Deployment completed successfully!"
echo "📊 Dashboard: http://grafana.${ENVIRONMENT}.example.com/d/reservoir-ai-dashboard"
echo "📚 API Docs: https://${SERVICE_URL}/docs"
