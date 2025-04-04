import requests
import typing
from typing import Optional, Dict

# Define the Open-WebUI Chat Completion API endpoint
api_url = "https://cllama.crefle.com/api/chat/completions"  # Chat Completion API endpoint
jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImYxODcwYWNhLTU0YTctNDUwOC1hYjA4LTkzZmZhMzkwY2Y2ZSJ9.whvAHzyQ8wPe8xBjKcvPuvfmWJ6E7LvlCkW0h_Lwyg4"  # Replace with your JWT token


def send_query(messages: Dict) -> Optional[str]:

    # Define the payload for the API request
    payload = {
        "model": "gemma3:27b",  # Specify the model to use
        "messages": messages
    }

    # Add the Authorization header with the JWT token
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }

    # Make the API request with headers
    response = requests.post(api_url, headers=headers, json=payload)

    # Check the response
    if response.status_code == 200:
        generated_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        print("Generated Text:", generated_text)
        return generated_text
    else:
        print("Error:", response.status_code, response.text)
        return None
            
# Example usage 

