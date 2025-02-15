import os
import subprocess
import json
import logging
import base64
import requests
import re
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_file(path: str) -> str:
    """Reads the content of a file."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except Exception as e:
        logging.error(f"Error reading file {path}: {e}")
        raise

def write_file(path: str, content: str) -> None:
    """Writes content to a file."""
    try:
        with open(path, 'w') as f:
            f.write(content)
    except Exception as e:
        logging.error(f"Error writing to file {path}: {e}")
        raise

def run_command(command: str) -> str:
    """Runs a shell command and returns the output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {command}, Error: {e.stderr}")
        raise Exception(f"Command failed: {command}, Error: {e.stderr}")

def call_llm(prompt: str) -> str:
    """Calls the LLM with the given prompt."""
    api_proxy_token = os.environ.get("AIPROXY_TOKEN")
    if not api_proxy_token:
        raise ValueError("AIPROXY_TOKEN environment variable not set.")

    url = "https://api.aiproxy.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_proxy_token}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "GPT-4o-Mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"LLM API call failed: {e}")
        raise Exception(f"LLM API call failed: {e}")

def encode_image(image_path: str) -> str:
    """Encodes an image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        raise

def extract_date_strings(text: str) -> List[str]:
    """Extracts date strings from text using regex."""
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        r'\d{2}\.\d{2}\.\d{4}', # DD.MM.YYYY
        r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', #D/M/YY, DD-MM-YYYY
        r'\w+ \d{1,2}, \d{4}', # Month Day, Year
        r'\d{1,2} \w+ \d{4}', # Day Month Year
    ]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    return dates

def create_directory(path: str) -> None:
    """Creates a directory if it doesn't exist."""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as e:
        logging.error(f"Error creating directory {path}: {e}")
        raise