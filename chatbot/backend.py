from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import openai
import mysql.connector
import os
import re
import requests  # Add this import for requests

app = FastAPI()

# Set your OpenAI API key here (or load from environment)
# It's better to use environment variables instead of hardcoding API keys
openai.api_key = os.environ.get("OPENAI_API_KEY", "your-default-api-key-here")

# MySQL database connection info
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "imindian",
    "database": "lab_management"
}

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", or "system"
    content: str

class ChatRequest(BaseModel):
    conversation: List[ChatMessage]
    excel_data: Optional[str] = None  # optional text summarizing the Excel file content

class ChatResponse(BaseModel):
    reply: str
    query_results: List[Dict[str, Any]] = []
    columns: List[str] = []

class PredictionRequest(BaseModel):
    message: str
    features: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    prediction: Optional[str] = None
    explanation: Optional[str] = None
    chat_response: Optional[str] = None
    error: Optional[str] = None

def run_sql_query(sql: str):
    """Executes SQL on the MySQL database and returns rows + column names."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
        except Exception as e:
            print("SQL Error:", e)
            rows = []
            col_names = []
        finally:
            cursor.close()
            conn.close()
        return rows, col_names
    except Exception as e:
        print("Database connection error:", e)
        return [], []

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Receives the conversation + optional excel_data, then:
    1) Prepend a system prompt about the database & excel analysis.
    2) Calls ChatGPT with the combined prompt & conversation.
    3) Searches the reply for a SQL snippet (```sql ... ```).
    4) Executes that SQL if found, returns results + the assistant's full reply.
    """
    # Build messages for ChatGPT
    system_prompt = """You are a helpful assistant for a lab management system.
        You can:
        1) Answer questions about the database (schema below) and optionally generate an SQL snippet (in triple backticks) (generate only the sql query nothing else is needed in the response.
        2) Analyze data from an uploaded Excel file (the user will provide text or summary from the file).

        Detailed Database Schema (MySQL):

        organizations:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - name VARCHAR(255) NOT NULL
          - created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        departments:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - name VARCHAR(255) NOT NULL
          - organization_id INT NOT NULL  (FK -> organizations.id)
          - created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        experiment_types:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - name VARCHAR(255) NOT NULL
          - created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        instruments:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - name VARCHAR(255) NOT NULL
          - description TEXT
          - organization_id INT NOT NULL  (FK -> organizations.id)
          - created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        users:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - username VARCHAR(255) NOT NULL
          - email VARCHAR(255) NOT NULL
          - password_bcrypted TEXT NOT NULL
          - phone VARCHAR(15)
          - role ENUM('super_admin','org_admin','staff') NOT NULL
          - organization_id INT (FK -> organizations.id)
          - created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        experiments:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - title VARCHAR(255) NOT NULL
          - objective TEXT
          - type_id INT (FK -> experiment_types.id)
          - start_date DATE
          - end_date DATE
          - department_id INT (FK -> departments.id)
          - created_by INT (FK -> users.id)
          - created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        experiment_parameters:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - name VARCHAR(255) NOT NULL
          - type_id INT (FK -> experiment_types.id)
          - is_predefined BOOLEAN

        experiment_conditions:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - experiment_id INT (FK -> experiments.id)
          - parameter_id INT (FK -> experiment_parameters.id)
          - value TEXT

        experiment_collaborators:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - experiment_id INT NOT NULL (FK -> experiments.id)
          - user_id INT NOT NULL (FK -> users.id)

        experiment_data_entries:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - experiment_id INT NOT NULL (FK -> experiments.id)
          - recorded_by INT NOT NULL (FK -> users.id)
          - raw_data JSON
          - uploaded_file VARCHAR(255)
          - created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        experiment_instruments:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - experiment_id INT NOT NULL (FK -> experiments.id)
          - instrument_id INT NOT NULL (FK -> instruments.id)

        data_visualizations:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - experiment_id INT NOT NULL (FK -> experiments.id)
          - plot_type ENUM('bar','line','scatter')
          - settings JSON
          - created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        eln_reports:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - experiment_id INT NOT NULL (FK -> experiments.id)
          - observations TEXT
          - raw_data JSON
          - visualizations JSON
          - conclusions TEXT
          - pdf_path VARCHAR(255)
          - created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        notifications:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - user_id INT NOT NULL (FK -> users.id)
          - message TEXT NOT NULL
          - is_read BOOLEAN DEFAULT FALSE
          - created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        qna:
          - id INT AUTO_INCREMENT PRIMARY KEY
          - organization_id INT NOT NULL (FK -> organizations.id)
          - submitted_by INT NOT NULL (FK -> users.id)
          - message TEXT NOT NULL
          - status ENUM('open','in_progress','resolved') DEFAULT 'open'
          - resolved_by INT (FK -> users.id)
          - created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        If you include SQL, always enclose it in triple backticks like:
        ```sql
        SELECT * FROM ...
        Make sure of this one thing: Use valid MySQL syntax (and make sure that the query is such that no foreign keys ore primary keys (NO IDs ONLY NAMES) are displayed only fetch data a layman would understand(make appropriate joins if required)). You can also analyze/summarize the Excel data if the user provides it. 
        3) if i ask for any kind of a navigation just show me this link "https://www.figma.com/proto/7Fz2xWXCRU0NK7k2Eax5ds/Nakya?node-id=338-912&starting-point-node-id=13%3A988"
        """
    messages = [{"role": "system", "content": system_prompt}]

    # If user provided Excel data, add it as system context
    if request.excel_data:
        messages.append({
            "role": "system",
            "content": f"User's Excel/CSV data:\n\n{request.excel_data}"
        })

    # Add the ongoing conversation
    for msg in request.conversation:
        messages.append({"role": msg.role, "content": msg.content})

    try:
        # Call OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3
        )

        assistant_reply = response["choices"][0]["message"]["content"]

        # Detect a SQL snippet in triple backticks: ```sql ... ```
        sql_pattern = r"```(?:sql)?\s*(.*?)\s*```"
        match = re.search(sql_pattern, assistant_reply, re.IGNORECASE|re.DOTALL)

        query_results = []
        columns = []
        if match:
            sql_code = match.group(1).strip()
            rows, col_names = run_sql_query(sql_code)
            query_results = [dict(zip(col_names, row)) for row in rows]
            columns = col_names

        return ChatResponse(
            reply=assistant_reply,
            query_results=query_results,
            columns=columns
        )
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return ChatResponse(
            reply=f"An error occurred: {str(e)}",
            query_results=[],
            columns=[]
        )

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    """
    Endpoint to handle predictions that may use an ML model.
    """
    try:
        user_message = request.message
        features = request.features

        # Step 1: Decide if ML model is needed
        use_ml = should_use_ml(user_message)

        if use_ml and features:
            ML_API_URL = os.environ.get("MODEL_API_URL", "http://localhost:5001/predict")
            # Step 2: Call the ML Model API
            ml_response = requests.post(ML_API_URL, json={"features": features})
            ml_result = ml_response.json()["prediction"]

            # Step 3: Ask ChatGPT to explain the ML prediction
            explanation_prompt = f"My model predicted {ml_result}. Can you explain what this means in a user-friendly way?"

            chatgpt_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": explanation_prompt}]
            )

            return PredictionResponse(
                prediction=ml_result,
                explanation=chatgpt_response["choices"][0]["message"]["content"]
            )

        else:
            # Step 4: If no ML model is needed, just use ChatGPT
            chatgpt_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": user_message}]
            )

            return PredictionResponse(
                chat_response=chatgpt_response["choices"][0]["message"]["content"]
            )

    except Exception as e:
        return PredictionResponse(error=str(e))

# Function to decide when to use the ML model
def should_use_ml(user_message):
    try:
        decision_prompt = f"""
        The user sent the following request: "{user_message}".
        Should I use a machine learning model to generate a prediction? 
        Respond with only 'YES' or 'NO'.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": decision_prompt}]
        )

        return response["choices"][0]["message"]["content"].strip().upper() == "YES"
    except Exception as e:
        print(f"Error in should_use_ml: {e}")
        return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)