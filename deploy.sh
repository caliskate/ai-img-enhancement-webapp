#!/bin/bash

echo "🚀 Deploying Image Enhancement API..."

# Pull latest code
git pull origin main

# Stop existing containers
docker-compose down

# Build new images
docker-compose build --no-cache

# Start services
docker-compose up -d

# Show logs
docker-compose logs -f
