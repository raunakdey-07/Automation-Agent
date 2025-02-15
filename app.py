import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import logging

from agent import AutomationAgent
from utils import read_file

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
agent = AutomationAgent()

@app.route('/run', methods=['POST'])
def run_task():
    """Executes a task based on the provided description."""
    task_description = request.args.get('task')
    if not task_description:
        return jsonify({"error": "Task description is required."}), 400

    try:
        agent.execute_task(task_description)
        return jsonify({"status": "Task executed successfully."}), 200
    except ValueError as e:
        logging.error(f"Task error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.exception(f"Agent error: {e}")
        return jsonify({"error": "Internal server error."}), 500

@app.route('/read', methods=['GET'])
def read_file_content():
    """Reads and returns the content of the specified file."""
    file_path = request.args.get('path')
    if not file_path:
        return jsonify({"error": "File path is required."}), 400

    try:
        content = read_file(file_path)
        return content, 200, {'Content-Type': 'text/plain'}
    except FileNotFoundError:
        return jsonify({"error": "File not found."}), 404
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return jsonify({"error": "Internal server error."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)