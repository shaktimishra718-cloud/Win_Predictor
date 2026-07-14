import time
import requests
def fetch_with_retry(url, headers, max_retries=5):
    """Fetches data with automatic backoff for 429 errors."""
    for i in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            # Exponential Backoff: Wait 6s, 7s, 9s, 13s, etc.
            wait_time = (2 ** i) + 5 
            print(f"429 hit! Waiting {wait_time}s before retrying...")
            time.sleep(wait_time)
        else:
            print(f"Request failed with status {response.status_code}")
            return None
    return None
