#!/bin/bash

# Deployment script for Google Cloud Run
# Make sure you're authenticated: gcloud auth login

set -e  # Exit on any error

# Configuration
PROJECT_ID=${PROJECT_ID:-"your-project-id"}
SERVICE_NAME="cigna-insurance-chatbot"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Starting deployment to Google Cloud Run..."
echo "Project: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t ${IMAGE_NAME}:latest .

echo "📤 Pushing image to Google Container Registry..."
docker push ${IMAGE_NAME}:latest

# Deploy to Cloud Run
echo "🚢 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_NAME}:latest \
    --region=${REGION} \
    --platform=managed \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=2 \
    --timeout=3600 \
    --concurrency=100 \
    --max-instances=10 \
    --set-env-vars=ENVIRONMENT=production \
    --project=${PROJECT_ID}

echo "✅ Deployment complete!"
echo "🌐 Your service will be available at the URL shown above."
echo ""
echo "📋 Useful commands:"
echo "  View logs: gcloud run logs tail ${SERVICE_NAME} --region=${REGION}"
echo "  Update service: gcloud run services update ${SERVICE_NAME} --region=${REGION}"
echo "  Delete service: gcloud run services delete ${SERVICE_NAME} --region=${REGION}"