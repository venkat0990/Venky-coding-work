Technical Assessment

This document outlines the development of microservice consisting of two main components: a UI backend service and an AI backend service.The UI end takes the image input from the user. The AI backend utilizes a lightweight open-source model to perform object detection and returns the results in a structured JSON format. The two components communicate seamlessly to provide a comprehensive solution to the user.

Prerequisites
To successfully replicate this solution, ensure you have the following installed:
Docker: For containerization of the application.
Python (Flask or FastAPI): For building the backend services.
Lightweight Detection Model: Any open-source model suitable for object detection (e.g., YOLO, MobileNet SSD,etc)

(for reference use:- https://github.com/ultralytics/yolov3) for detection, if you dont have GPU, use CPU to complete your task
Deliverables
Your task project folder should be zipped and mailed to us,such that we should be able to replicate your solution in our system or share your github private link. 
A documentation to your project solution, essentially the steps, how did you to reach to the solution (any reference you took from anywhere mention it in the documentation)
Output images (containing the bounding box) and corresponding json files.

