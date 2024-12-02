'''Prerequisites and Setup
Install Docker: Follow the Docker installation guide to install Docker.

Install Python & Pip: Ensure Python (>=3.9) and pip are installed.

Install dependencies: Create requirements.txt files for each service with necessary dependencies. For example:

requirements.txt for UI Backend:
Flask
requests

requirements.txt for AI Backend:
Flask
tensorflow
opencv-python

Model: Download or train a lightweight object detection model (e.g., MobileNet SSD) and store it in a directory.
In the AI backend code, point to the directory where the model is saved (e.g., saved_model_path).


Running the Solution

Build and Run the Docker Containers:'''
docker-compose up --build

'''Access the UI: Navigate to http://localhost:5000 to upload an image and see object detection results.

Conclusion
This microservice architecture integrates a UI backend for image handling with an AI backend for object detection using a lightweight model.
Docker provides a streamlined way to manage and deploy both services, ensuring that the entire solution runs seamlessly in a containerized environment.
