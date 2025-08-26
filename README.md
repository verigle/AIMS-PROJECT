# AI For Science Master project 
---

[![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](#)
[![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?logo=visual-studio-code&logoColor=white)](#)
[![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)](#)
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](#)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-4285F4?logo=googlecloud&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](#)
![Awesome](https://img.shields.io/badge/Awesome-ffd700?logo=awesome&logoColor=black)

---

![Project Screenshot](images/aurora.jpg)


# Overview

This project is about getying accurate weather forecast with minimal cost using PEFT techniques like LoRA, to improve an AI Weather Foundation Model, here Microsoft-Aurora.


# How to run?
### STEPS:


Clone the repository

```bash
https://github.com/koomited/cowrywise-customer-plan-abandonment.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```
Now,
```bash
open up you local host and port
```


In Case you want to try training
```bash
# run the containers to create the database and tables
docker compose up -d
```
Then 
```bash
# run the main.py
python main.py
```




## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/koomi/cowrywise-customer-plan-abandonment.mlflow
MLFLOW_TRACKING_USERNAME=koomi \
MLFLOW_TRACKING_PASSWORD=b918e1b0fdebad41476188587e7a2996300c6ee2\
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/koomi/cowrywise-customer-plan-abandonment.mlflow

export MLFLOW_TRACKING_USERNAME=koomi

export MLFLOW_TRACKING_PASSWORD=b918e1b0fdebad41476188587e7a2996300c6ee2

```
Make sure you replace the credentials by yours.

# DockerHub CI/CD Workflow

This GitHub Actions workflow automates the build and push of a Docker image to Docker Hub whenever code is pushed to the `main` branch.

## Workflow steps

1. Checkout the repository code.
2. Log in to Docker Hub using secrets.
3. Build a Docker image tagged as `your-dockerhub-username/cowrywise-customer-plan-abandonment:latest`.
4. Push the Docker image to Docker Hub.

## Setup

- Create a Docker Hub repository named `cowrywise-customer-plan-abandonment`.
- Add the following GitHub Secrets in your repository settings:
  - `DOCKERHUB_USERNAME`: Your Docker Hub username.
  - `DOCKERHUB_PASSWORD`: Your Docker Hub password or access token.

## Usage

Push changes to the `main` branch to trigger this workflow automatically.

## Run the deploy image locally

To run the Docker image locally or on your server, use this command:

```bash
docker run -d --name plan-abandonment -p 8080:8080 tousside/cowrywise-customer-plan-abandonment:latest
```
```bash
Acess the UI on your browser using localhost:8080
```

# GCP CI/CD Deployment with GitHub Actions

This README provides step-by-step instructions for setting up CI/CD deployment on Google Cloud Platform using GitHub Actions.

## 1. Login to Google Cloud Console

## 2. Create Service Account for deployment
#with specific access
1. Compute Engine access: For virtual machine management
2. Artifact Registry: To save your docker image in GCP
3. Cloud Run: For containerized application deployment (optional)

#Description: About the deployment
1. Build docker image of the source code
2. Push your docker image to Artifact Registry
3. Launch Your Compute Engine VM
4. Pull Your image from Artifact Registry in VM
5. Launch your docker image in VM

#Required Roles:
1. Artifact Registry Administrator
2. Compute Admin
3. Service Account User
4. Storage Admin

## 3. Create Artifact Registry repository to store/save docker image
- Navigate to Artifact Registry > Create Repository
- Choose Docker format
- Save the repository URL: `REGION-docker.pkg.dev/PROJECT-ID/REPO-NAME`
- Example: `asia-south1-docker.pkg.dev/my-project-12345/mlproj`

## 4. Create Compute Engine VM (Ubuntu)
- Choose appropriate machine type
- Allow HTTP/HTTPS traffic
- Note the external IP address

## 5. Open VM and Install docker in Compute Engine:
#optional
```bash
sudo apt-get update -y
sudo apt-get upgrade -y
```

#required
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

#Install Google Cloud CLI
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

## 6. Configure VM as self-hosted runner:
settings > actions > runner > new self hosted runner > choose os > then run command one by one

## 7. Setup GitHub secrets:
```
GCP_PROJECT_ID=your-project-id
GCP_SA_KEY=<service-account-json-key-base64-encoded>
GCP_REGION=asia-south1
ARTIFACT_REGISTRY_URL=asia-south1-docker.pkg.dev/your-project-id
REPOSITORY_NAME=mlproj
VM_INSTANCE_NAME=your-vm-name
VM_ZONE=asia-south1-a
```

## 8. Authentication setup:
- Create service account key (JSON format)
- Base64 encode the JSON key: `cat key.json | base64`
- Add the encoded key to GitHub secrets as `GCP_SA_KEY`

## 9. Configure Artifact Registry authentication on VM:
```bash
gcloud auth configure-docker asia-south1-docker.pkg.dev
```


##  Future works
- For now, I have not tested the deployement because of lack of credit on gcp. But I will do it very soon.
- Automate model monitoring and retrain model if performance decreased.
- Flask backend is modular and easy to expand (e.g., adding new routes or model versions).
- UI can be enhanced or replaced with a React or Vue frontend if needed.