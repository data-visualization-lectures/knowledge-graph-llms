import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("GOOGLE_API_KEY not found in environment variables.")
    api_key = input("Please enter your Google API Key: ").strip()

if not api_key:
    print("Error: API Key is required.")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    print("\nAvailable Models:")
    found = False
    if 'models' in data:
        for model in data['models']:
            if 'generateContent' in model.get('supportedGenerationMethods', []):
                print(f"- {model['name']}")
                found = True
    
    if not found:
        print("No models found with 'generateContent' capability.")
        
except Exception as e:
    print(f"Error listing models: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response: {e.response.text}")
