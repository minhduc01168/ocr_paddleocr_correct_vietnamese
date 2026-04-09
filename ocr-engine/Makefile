# Cấu hình AWS
AWS_REGION=us-east-1
ECR_REPO_NAME=aihub-ocr-engine
AWS_ACCOUNT_ID=$(shell aws sts get-caller-identity --query Account --output text)
ECR_URL=$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
IMAGE_TAG=latest

# Tên Image
IMAGE_NAME=$(ECR_REPO_NAME)

.PHONY: build push run login help

help:
	@echo "Lệnh hữu ích cho OCR Engine:"
	@echo "  make build          - Build Docker image cho Engine"
	@echo "  make run            - Chạy Engine tại Local (Port 8002)"
	@echo "  make login          - Đăng nhập vào AWS ECR"
	@echo "  make push           - Push image lên AWS ECR để triển khai SageMaker"

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -p 8002:8002 --env API_PORT=8002 $(IMAGE_NAME)

login:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_URL)

push: login
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(ECR_URL)/$(ECR_REPO_NAME):$(IMAGE_TAG)
	docker push $(ECR_URL)/$(ECR_REPO_NAME):$(IMAGE_TAG)
	@echo "[+] Image đã được đẩy lên: $(ECR_URL)/$(ECR_REPO_NAME):$(IMAGE_TAG)"
