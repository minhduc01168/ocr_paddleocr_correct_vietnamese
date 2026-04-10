# ── OCR Service — Makefile ───────────────────────────────────────────────
# Build & Run Helpers for Local Docker and SageMaker AWS

SERVICE_NAME=aihub-ocr-service
IMAGE_TAG=latest
ECR_REPO_URL=[YOU_AWS_ACCOUNT_ID].dkr.ecr.[AWS_REGION].amazonaws.com/$(SERVICE_NAME)

.PHONY: help build run stop push logs

help:
	@echo "Available commands:"
	@echo "  build   - Build Docker image localy"
	@echo "  run     - Run Docker image on port 8080"
	@echo "  stop    - Stop the running container"
	@echo "  push    - Login, tag and push to Amazon ECR"
	@echo "  logs    - View container logs"

build:
	docker build -t $(SERVICE_NAME):$(IMAGE_TAG) .

run:
	docker run -d \
		--name $(SERVICE_NAME) \
		-p 8001:8001 \
		--env-file .env \
		$(SERVICE_NAME):$(IMAGE_TAG)

stop:
	docker stop $(SERVICE_NAME) || true
	docker rm $(SERVICE_NAME) || true

logs:
	docker logs -f $(SERVICE_NAME)

# ── AWS Deployment ────────────────────────────────────────────────────────

push:
	aws ecr get-login-password --region [AWS_REGION] | docker login --username AWS --password-stdin $(ECR_REPO_URL)
	docker tag $(SERVICE_NAME):$(IMAGE_TAG) $(ECR_REPO_URL):$(IMAGE_TAG)
	docker push $(ECR_REPO_URL):$(IMAGE_TAG)
	@echo "Done! Image $(ECR_REPO_URL):$(IMAGE_TAG) is now available in ECR."
