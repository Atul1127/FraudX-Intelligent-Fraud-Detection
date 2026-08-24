# AWS Deployment

FraudX is prepared for an AWS ECS Fargate deployment.

## Architecture

GitHub Actions -> Amazon ECR -> ECS Fargate -> Application Load Balancer

MongoDB and MLflow are intentionally kept out of the first production deployment. The current local Docker Compose stack remains the development/integration environment. For production, use a managed MongoDB deployment and a durable MLflow tracking/artifact backend.

## Prerequisites

- AWS account
- AWS CLI configured
- Terraform >= 1.6
- GitHub repository secrets:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_REGION`

## Terraform

From this directory:

```bash
terraform init
terraform plan -var='aws_region=ap-south-1'
terraform apply -var='aws_region=ap-south-1'
```

The Terraform stack creates an ECR repository, ECS cluster, task execution role, task definition, security groups, and an internet-facing Application Load Balancer.

After infrastructure exists, GitHub Actions can build and push the API image to ECR and update the ECS service.
