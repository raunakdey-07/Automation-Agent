import os
import json
import logging
from datetime import datetime
import dateutil.parser
import sqlite3
import requests
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils import read_file, write_file, run_command, call_llm, encode_image, extract_date_strings, create_directory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AutomationAgent:
    def __init__(self):
        self.data_dir = "/data"
        create_directory(self.data_dir)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the SentenceTransformer model

    def execute_task(self, task_description: str) -> None:
        """Executes the given task description."""
        logging.info(f"Executing task: {task_description}")

        try:
            if "install uv" in task_description.lower() and "datagen.py" in task_description.lower():
                self._handle_a1(task_description)
            elif "format the contents of" in task_description.lower() and ".md" in task_description.lower():
                self._handle_a2(task_description)
            elif "count the number of wednesdays" in task_description.lower():
                self._handle_a3(task_description)
            elif "sort the array of contacts" in task_description.lower():
                self._handle_a4(task_description)
            elif "most recent .log file" in task_description.lower():
                self._handle_a5(task_description)
            elif "extract the first occurrance of each h1" in task_description.lower():
                self._handle_a6(task_description)
            elif "extract the sender’s email address" in task_description.lower():
                self._handle_a7(task_description)
            elif "extract the card number" in task_description.lower():
                self._handle_a8(task_description)
            elif "most similar pair of comments" in task_description.lower():
                self._handle_a9(task_description)
            elif "total sales of all the items in the “gold” ticket type" in task_description.lower():
                self._handle_a10(task_description)
            elif "fetch data from an api and save it" in task_description.lower():
                self._handle_b3(task_description)
            elif "clone a git repo and make a commit" in task_description.lower():
                self._handle_b4(task_description)
            elif "run a sql query on a sqlite or duckdb database" in task_description.lower():
                self._handle_b5(task_description)
            elif "extract data from (i.e. scrape) a website" in task_description.lower():
                self._handle_b6(task_description)
            elif "compress or resize an image" in task_description.lower():
                self._handle_b7(task_description)
            elif "transcribe audio from an mp3 file" in task_description.lower():
                self._handle_b8(task_description)
            elif "convert markdown to html" in task_description.lower():
                self._handle_b9(task_description)
            elif "write an api endpoint that filters a csv file and returns json data" in task_description.lower():
                self._handle_b10(task_description)
            else:
                raise ValueError("Task not supported.")

        except Exception as e:
            logging.error(f"Task execution failed: {e}")
            raise

    def _handle_a1(self, task_description: str) -> None:
        """A1. Install uv (if required) and run datagen.py with ${user.email}."""
        try:
            user_email = os.environ.get("USER_EMAIL", "test@example.com")  # Default email
            datagen_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"

            # Download datagen.py
            command = f"curl -o /tmp/datagen.py {datagen_url}"
            run_command(command)

            # Install uv if not already installed
            try:
                run_command("uv --version")
            except:
                run_command("pip install uv")

            # Run datagen.py with uv
            command = f"uv python /tmp/datagen.py {user_email}"
            run_command(command)

            logging.info("A1 completed successfully.")

        except Exception as e:
            logging.error(f"A1 failed: {e}")
            raise

    def _handle_a2(self, task_description: str) -> None:
        """A2. Format the contents of /data/format.md using prettier@3.4.2."""
        file_path = os.path.join(self.data_dir, "format.md")
        try:
            # Install prettier if not already installed
            try:
                run_command("prettier --version")
            except:
                run_command("npm install -g prettier@3.4.2")

            # Format the file using prettier
            command = f"prettier --write {file_path}"
            run_command(command)

            logging.info("A2 completed successfully.")

        except Exception as e:
            logging.error(f"A2 failed: {e}")
            raise

    def _handle_a3(self, task_description: str) -> None:
        """A3. Count the number of Wednesdays in /data/dates.txt and write to /data/dates-wednesdays.txt."""
        input_file_path = os.path.join(self.data_dir, "dates.txt")
        output_file_path = os.path.join(self.data_dir, "dates-wednesdays.txt")

        try:
            content = read_file(input_file_path)
            date_strings = extract_date_strings(content)

            wednesday_count = 0
            for date_str in date_strings:
                try:
                    date_obj = dateutil.parser.parse(date_str)
                    if date_obj.weekday() == 2:  # Wednesday is 2
                        wednesday_count += 1
                except ValueError:
                    logging.warning(f"Invalid date format: {date_str}")
                    continue

            write_file(output_file_path, str(wednesday_count))
            logging.info("A3 completed successfully.")

        except Exception as e:
            logging.error(f"A3 failed: {e}")
            raise

    def _handle_a4(self, task_description: str) -> None:
        """A4. Sort the array of contacts in /data/contacts.json by last_name, then first_name."""
        input_file_path = os.path.join(self.data_dir, "contacts.json")
        output_file_path = os.path.join(self.data_dir, "contacts-sorted.json")

        try:
            content = read_file(input_file_path)
            contacts = json.loads(content)

            sorted_contacts = sorted(contacts, key=lambda x: (x.get('last_name', ''), x.get('first_name', '')))

            write_file(output_file_path, json.dumps(sorted_contacts, indent=4))
            logging.info("A4 completed successfully.")

        except Exception as e:
            logging.error(f"A4 failed: {e}")
            raise

    def _handle_a5(self, task_description: str) -> None:
        """A5. Write the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt, most recent first."""
        logs_dir = os.path.join(self.data_dir, "logs")
        output_file_path = os.path.join(self.data_dir, "logs-recent.txt")

        try:
            log_files = [f for f in os.listdir(logs_dir) if f.endswith(".log")]
            log_files_with_path = [os.path.join(logs_dir, f) for f in log_files]
            log_files_with_time = [(f, os.path.getmtime(f)) for f in log_files_with_path]
            sorted_log_files = sorted(log_files_with_time, key=lambda x: x[1], reverse=True)[:10]

            first_lines = []
            for file_path, _ in sorted_log_files:
                try:
                    with open(file_path, 'r') as f:
                        first_line = f.readline().strip()
                        first_lines.append(first_line)
                except Exception as e:
                    logging.warning(f"Could not read first line from {file_path}: {e}")
                    first_lines.append(f"Could not read {os.path.basename(file_path)}")

            write_file(output_file_path, "\n".join(first_lines))
            logging.info("A5 completed successfully.")

        except Exception as e:
            logging.error(f"A5 failed: {e}")
            raise

    def _handle_a6(self, task_description: str) -> None:
        """A6. Find all Markdown files, extract the first H1, and create an index file."""
        docs_dir = os.path.join(self.data_dir, "docs")
        index_file_path = os.path.join(docs_dir, "index.json")

        try:
            index = {}
            for filename in os.listdir(docs_dir):
                if filename.endswith(".md"):
                    file_path = os.path.join(docs_dir, filename)
                    content = read_file(file_path)
                    # Extract the first H1 using regex
                    match = re.search(r"^#\s+(.*)", content, re.MULTILINE)
                    if match:
                        title = match.group(1).strip()
                        index[filename] = title
                    else:
                        index[filename] = None

            write_file(index_file_path, json.dumps(index, indent=4))
            logging.info("A6 completed successfully.")

        except Exception as e:
            logging.error(f"A6 failed: {e}")
            raise

    def _handle_a7(self, task_description: str) -> None:
        """A7. Extract sender's email address from /data/email.txt using LLM."""
        email_file_path = os.path.join(self.data_dir, "email.txt")
        output_file_path = os.path.join(self.data_dir, "email-sender.txt")

        try:
            email_content = read_file(email_file_path)
            prompt = f"Extract the sender's email address from the following email:\n{email_content}\n\nSender's Email:"
            sender_email = call_llm(prompt)

            # Basic email validation (you can improve this)
            if "@" not in sender_email:
                raise ValueError(f"Invalid email format: {sender_email}")

            write_file(output_file_path, sender_email)
            logging.info("A7 completed successfully.")

        except Exception as e:
            logging.error(f"A7 failed: {e}")
            raise

    def _handle_a8(self, task_description: str) -> None:
        """A8. Extract credit card number from /data/credit-card.png using LLM."""
        image_path = os.path.join(self.data_dir, "credit-card.png")
        output_file_path = os.path.join(self.data_dir, "credit-card.txt")

        try:
            base64_image = encode_image(image_path)
            prompt = f"Extract the credit card number from the following image:\n\n<image>{base64_image}</image>\n\nCredit Card Number (no spaces):"
            card_number = call_llm(prompt)

            # Remove spaces from the extracted card number
            card_number = card_number.replace(" ", "")

            # Basic card number validation (you can improve this)
            if not card_number.isdigit():
                raise ValueError(f"Invalid card number format: {card_number}")

            write_file(output_file_path, card_number)
            logging.info("A8 completed successfully.")

        except Exception as e:
            logging.error(f"A8 failed: {e}")
            raise

    def _handle_a9(self, task_description: str) -> None:
        """A9. Find the most similar pair of comments in /data/comments.txt using embeddings."""
        comments_file_path = os.path.join(self.data_dir, "comments.txt")
        output_file_path = os.path.join(self.data_dir, "comments-similar.txt")

        try:
            comments = read_file(comments_file_path).splitlines()
            comments = [comment.strip() for comment in comments if comment.strip()]  # Remove empty comments

            if len(comments) < 2:
                raise ValueError("Not enough comments to compare.")

            embeddings = self.model.encode(comments)

            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings)

            # Find the most similar pair (excluding self-similarity)
            np.fill_diagonal(similarity_matrix, -1)  # Ignore self-similarity
            max_index = np.argmax(similarity_matrix)
            comment1_index, comment2_index = np.unravel_index(max_index, similarity_matrix.shape)

            similar_comments = [comments[comment1_index], comments[comment2_index]]

            write_file(output_file_path, "\n".join(similar_comments))
            logging.info("A9 completed successfully.")

        except Exception as e:
            logging.error(f"A9 failed: {e}")
            raise

    def _handle_a10(self, task_description: str) -> None:
        """A10. Calculate total sales of "Gold" tickets from /data/ticket-sales.db."""
        db_path = os.path.join(self.data_dir, "ticket-sales.db")
        output_file_path = os.path.join(self.data_dir, "ticket-sales-gold.txt")

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
            result = cursor.fetchone()[0]

            total_sales = result if result is not None else 0

            write_file(output_file_path, str(total_sales))
            conn.close()
            logging.info("A10 completed successfully.")

        except Exception as e:
            logging.error(f"A10 failed: {e}")
            raise

    def _handle_b3(self, task_description: str) -> None:
        """B3. Fetch data from an API and save it."""
        try:
            # Extract API URL and file path from the task description using LLM
            prompt = f"Extract the API URL and the file path to save the data to from the following task description:\n{task_description}\n\nOutput in JSON format: {{\"api_url\": \"<api_url>\", \"file_path\": \"<file_path>\"}}"
            llm_response = call_llm(prompt)
            try:
                extracted_info = json.loads(llm_response)
                api_url = extracted_info.get("api_url")
                file_path = extracted_info.get("file_path")
            except (json.JSONDecodeError, AttributeError):
                raise ValueError(f"Could not extract API URL and file path from LLM response: {llm_response}")

            if not api_url or not file_path:
                raise ValueError("API URL and file path must be provided.")

            # Ensure the file path is within the /data directory
            if not file_path.startswith(self.data_dir):
                raise ValueError("File path must be within the /data directory.")

            # Fetch data from the API
            response = requests.get(api_url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.text

            # Save the data to the specified file
            write_file(file_path, data)
            logging.info("B3 completed successfully.")

        except Exception as e:
            logging.error(f"B3 failed: {e}")
            raise

    def _handle_b4(self, task_description: str) -> None:
        """B4. Clone a git repo and make a commit."""
        try:
            # Extract repo URL and commit message from the task description using LLM
            prompt = f"Extract the repository URL and the commit message from the following task description:\n{task_description}\n\nOutput in JSON format: {{\"repo_url\": \"<repo_url>\", \"commit_message\": \"<commit_message>\"}}"
            llm_response = call_llm(prompt)
            try:
                extracted_info = json.loads(llm_response)
                repo_url = extracted_info.get("repo_url")
                commit_message = extracted_info.get("commit_message")
            except (json.JSONDecodeError, AttributeError):
                raise ValueError(f"Could not extract repository URL and commit message from LLM response: {llm_response}")

            if not repo_url or not commit_message:
                raise ValueError("Repository URL and commit message must be provided.")

            # Define the clone directory within /data
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            clone_dir = os.path.join(self.data_dir, repo_name)

            # Clone the repository
            if not os.path.exists(clone_dir):
                command = f"git clone {repo_url} {clone_dir}"
                run_command(command)

            # Create a dummy file to commit
            dummy_file = os.path.join(clone_dir, "dummy.txt")
            write_file(dummy_file, "This is a dummy file.")

            # Configure git (required for committing)
            run_command(f"git config --global user.email 'you@example.com'")
            run_command(f"git config --global user.name 'Your Name'")

            # Add, commit, and push the changes
            command = f"cd {clone_dir} && git add . && git commit -m '{commit_message}' && git push"
            run_command(command)

            logging.info("B4 completed successfully.")

        except Exception as e:
            logging.error(f"B4 failed: {e}")
            raise

    def _handle_b5(self, task_description: str) -> None:
        """B5. Run a SQL query on a SQLite or DuckDB database."""
        try:
            # Extract database path, query, and output file path from the task description using LLM
            prompt = f"Extract the database path, SQL query, and output file path from the following task description:\n{task_description}\n\nOutput in JSON format: {{\"db_path\": \"<db_path>\", \"query\": \"<query>\", \"output_file\": \"<output_file>\"}}"
            llm_response = call_llm(prompt)
            try:
                extracted_info = json.loads(llm_response)
                db_path = extracted_info.get("db_path")
                query = extracted_info.get("query")
                output_file = extracted_info.get("output_file")
            except (json.JSONDecodeError, AttributeError):
                raise ValueError(f"Could not extract database path, query, and output file path from LLM response: {llm_response}")

            if not db_path or not query or not output_file:
                raise ValueError("Database path, query, and output file path must be provided.")

            # Ensure the database path and output file are within the /data directory
            if not db_path.startswith(self.data_dir) or not output_file.startswith(self.data_dir):
                raise ValueError("Database path and output file must be within the /data directory.")

            # Determine the database type (SQLite or DuckDB) based on the file extension
            if db_path.endswith(".db") or db_path.endswith(".sqlite"):
                db_type = "sqlite"
            elif db_path.endswith(".duckdb"):
                db_type = "duckdb"
            else:
                raise ValueError("Unsupported database type. Only SQLite (.db, .sqlite) and DuckDB (.duckdb) are supported.")

            # Execute the SQL query based on the database type
            if db_type == "sqlite":
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                conn.close()
            elif db_type == "duckdb":
                import duckdb
                conn = duckdb.connect(database=db_path, read_only=True)
                results = conn.execute(query).fetchall()
                conn.close()

            # Format the results as a list of dictionaries
            column_names = [description[0] for description in cursor.description]
            formatted_results = [dict(zip(column_names, row)) for row in results]

            # Write the results to the output file as JSON
            write_file(output_file, json.dumps(formatted_results, indent=4))
            logging.info("B5 completed successfully.")

        except Exception as e:
            logging.error(f"B5 failed: {e}")
            raise

    def _handle_b6(self, task_description: str) -> None:
        """B6. Extract data from (i.e. scrape) a website."""
        try:
            # Extract URL and output file path from the task description using LLM
            prompt = f"Extract the URL of the website to scrape and the output file path from the following task description:\n{task_description}\n\nOutput in JSON format: {{\"url\": \"<url>\", \"output_file\": \"<output_file>\"}}"
            llm_response = call_llm(prompt)
            try:
                extracted_info = json.loads(llm_response)
                url = extracted_info.get("url")
                output_file = extracted_info.get("output_file")
            except (json.JSONDecodeError, AttributeError):
                raise ValueError(f"Could not extract URL and output file path from LLM response: {llm_response}")

            if not url or not output_file:
                raise ValueError("URL and output file path must be provided.")

            # Ensure the output file is within the /data directory
            if not output_file.startswith(self.data_dir):
                raise ValueError("Output file must be within the /data directory.")

            # Fetch the website content
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            html_content = response.text

            # Write the extracted text to the output file
            write_file(output_file, html_content)
            logging.info("B6 completed successfully.")

        except Exception as e:
            logging.error(f"B6 failed: {e}")
            raise

    def _handle_b7(self, task_description: str) -> None:
        """B7. Compress or resize an image."""
        try:
            # Extract image path, output path, and operation (compress/resize) from the task description using LLM
            prompt = f"Extract the image path, output path, and operation (compress/resize) from the following task description:\n{task_description}\n\nOutput in JSON format: {{\"image_path\": \"<image_path>\", \"output_path\": \"<output_path>\", \"operation\": \"<operation>\", \"width\": <width>, \"height\": <height>, \"quality\": <quality>}}"
            llm_response = call_llm(prompt)
            try:
                extracted_info = json.loads(llm_response)
                image_path = extracted_info.get("image_path")
                output_path = extracted_info.get("output_path")
                operation = extracted_info.get("operation")
                width = extracted_info.get("width")
                height = extracted_info.get("height")
                quality = extracted_info.get("quality")
            except (json.JSONDecodeError, AttributeError):
                raise ValueError(f"Could not extract image path, output path, and operation from LLM response: {llm_response}")

            if not image_path or not output_path or not operation:
                raise ValueError("Image path, output path, and operation must be provided.")

            # Ensure the image path and output path are within the /data directory
            if not image_path.startswith(self.data_dir) or not output_path.startswith(self.data_dir):
                raise ValueError("Image path and output path must be within the /data directory.")

            # Open the image using Pillow
            img = Image.open(image_path)

            # Perform the specified operation
            if operation.lower() == "resize":
                if not width or not height:
                    raise ValueError("Width and height must be provided for resize operation.")
                img = img.resize((width, height))
            elif operation.lower() == "compress":
                if not quality:
                    quality = 85  # Default quality
                img = img.convert('RGB')  # Ensure image is in RGB format for JPEG compression
                img.save(output_path, format='JPEG', quality=quality)
                logging.info("B7 (compress) completed successfully.")
                return
            else:
                raise ValueError("Unsupported operation. Only 'resize' and 'compress' are supported.")

            # Save the modified image to the output path
            img.save(output_path)
            logging.info("B7 completed successfully.")

        except Exception as e:
            logging.error(f"B7 failed: {e}")
            raise

    def _handle_b8(self, task_description: str) -> None:
        """B8. Transcribe audio from an MP3 file."""
        try:
            # Extract audio file path and output file path from the task description using LLM
            prompt = f"Extract the audio file path and output file path from the following task description:\n{task_description}\n\nOutput in JSON format: {{\"audio_path\": \"<audio_path>\", \"output_path\": \"<output_path>\"}}"
            llm_response = call_llm(prompt)
            try:
                extracted_info = json.loads(llm_response)
                audio_path = extracted_info.get("audio_path")
                output_path = extracted_info.get("output_path")
            except (json.JSONDecodeError, AttributeError):
                raise ValueError(f"Could not extract audio file path and output file path from LLM response: {llm_response}")

            if not audio_path or not output_path:
                raise ValueError("Audio file path and output file path must be provided.")

            # Ensure the audio file path and output path are within the /data directory
            if not audio_path.startswith(self.data_dir) or not output_path.startswith(self.data_dir):
                raise ValueError("Audio file path and output path must be within the /data directory.")

            # Use ffmpeg to convert the MP3 file to WAV format (required by some transcription tools)
            wav_path = audio_path.replace(".mp3", ".wav")
            command = f"ffmpeg -i {audio_path} -acodec pcm_s16le -ac 1 -ar 16000 {wav_path}"
            run_command(command)

            # Transcribe the audio using a transcription tool (e.g., Whisper)
            try:
                import whisper
                model = whisper.load_model("base")  # You can choose different model sizes
                result = model.transcribe(wav_path)
                transcription = result["text"]
            except ImportError:
                raise ImportError("Whisper library not found. Please install it: pip install openai-whisper")
            except Exception as e:
                raise Exception(f"Transcription failed: {e}")

            # Write the transcription to the output file
            write_file(output_path, transcription)
            logging.info("B8 completed successfully.")

        except Exception as e:
            logging.error(f"B8 failed: {e}")
            raise

    def _handle_b9(self, task_description: str) -> None:
        """B9. Convert Markdown to HTML."""
        try:
            # Extract Markdown file path and output HTML file path from the task description using LLM
            prompt = f"Extract the Markdown file path and output HTML file path from the following task description:\n{task_description}\n\nOutput in JSON format: {{\"markdown_path\": \"<markdown_path>\", \"html_path\": \"<html_path>\"}}"
            llm_response = call_llm(prompt)
            try:
                extracted_info = json.loads(llm_response)
                markdown_path = extracted_info.get("markdown_path")
                html_path = extracted_info.get("html_path")
            except (json.JSONDecodeError, AttributeError):
                raise ValueError(f"Could not extract Markdown file path and output HTML file path from LLM response: {llm_response}")

            if not markdown_path or not html_path:
                raise ValueError("Markdown file path and output HTML file path must be provided.")

            # Ensure the Markdown file path and output HTML file path are within the /data directory
            if not markdown_path.startswith(self.data_dir) or not html_path.startswith(self.data_dir):
                raise ValueError("Markdown file path and output HTML file path must be within the /data directory.")

            # Read the Markdown content from the file
            markdown_content = read_file(markdown_path)

            # Convert Markdown to HTML using a library (e.g., markdown2)
            try:
                import markdown2
                html_content = markdown2.markdown(markdown_content)
            except ImportError:
                raise ImportError("markdown2 library not found. Please install it: pip install markdown2")
            except Exception as e:
                raise Exception(f"Markdown to HTML conversion failed: {e}")

            # Write the HTML content to the output file
            write_file(html_path, html_content)
            logging.info("B9 completed successfully.")

        except Exception as e:
            logging.error(f"B9 failed: {e}")
            raise

    def _handle_b10(self, task_description: str) -> None:
        """B10. Write an API endpoint that filters a CSV file and returns JSON data."""
        try:
            # Extract CSV file path, filter column, filter value, and API endpoint path from the task description using LLM
            prompt = f"Extract the CSV file path, filter column, filter value, and API endpoint path from the following task description:\n{task_description}\n\nOutput in JSON format: {{\"csv_path\": \"<csv_path>\", \"filter_column\": \"<filter_column>\", \"filter_value\": \"<filter_value>\", \"api_endpoint\": \"<api_endpoint>\"}}"
            llm_response = call_llm(prompt)
            try:
                extracted_info = json.loads(llm_response)
                csv_path = extracted_info.get("csv_path")
                filter_column = extracted_info.get("filter_column")
                filter_value = extracted_info.get("filter_value")
                api_endpoint = extracted_info.get("api_endpoint")
            except (json.JSONDecodeError, AttributeError):
                raise ValueError(f"Could not extract CSV file path, filter column, filter value, and API endpoint path from LLM response: {llm_response}")

            if not csv_path or not filter_column or not filter_value or not api_endpoint:
                raise ValueError("CSV file path, filter column, filter value, and API endpoint path must be provided.")

            # Ensure the CSV file path is within the /data directory
            if not csv_path.startswith(self.data_dir):
                raise ValueError("CSV file path must be within the /data directory.")

            # Read the CSV file and filter the data
            import csv
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                filtered_data = [row for row in reader if row[filter_column] == filter_value]

            # Return the filtered data as a string
            return json.dumps(filtered_data)
            logging.info("B10 completed successfully.")

        except Exception as e:
            logging.error(f"B10 failed: {e}")
            raise
