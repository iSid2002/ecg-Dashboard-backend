# ECG Analysis Dashboard

A full-stack application for ECG signal analysis and heart failure prediction.

## Project Structure

- `frontend/`: React frontend application
- `backend/`: FastAPI backend application

## Setup Instructions

1. Clone the repository
2. Navigate to the project directory
3. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

## Running the Application

1. Start the backend server:
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn app.main:app --reload
   ```

2. Start the frontend development server:
   ```bash
   cd frontend
   npm start
   ```

3. Open your browser and navigate to http://localhost:3000

## Features

- Generate synthetic ECG signals
- Train heart failure prediction models
- View ECG plots and analysis results
- Mobile-responsive design
