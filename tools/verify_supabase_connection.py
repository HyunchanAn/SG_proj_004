import toml
from supabase import create_client, Client
import sys

def verify():
    try:
        secrets = toml.load(".streamlit/secrets.toml")
        url = secrets["supabase"]["url"]
        key = secrets["supabase"]["key"]
        
        print(f"Testing connection to: {url}")
        print(f"Using key starting with: {key[:5]}...")
        
        supabase: Client = create_client(url, key)
        
        # Determine if we can make a simple request. 
        # Auth usually doesn't require a network request just to `create_client`
        # But we can try to get the session or something similar, or just check if client init didn't raise
        
        print("Supabase client initialized successfully.")
        
        # Try a simple "health check" or invalid login to see if it reaches the server
        try:
            # Attempt to sign in with a clearly fake user to check network/key validity
            supabase.auth.sign_in_with_password({"email": "test@example.com", "password": "wrongpassword"})
        except Exception as e:
            # We expect a login failure, but not a "connection refused" or "invalid key" error if possible.
            # However, supabase-py might raise specific exceptions map to API errors.
            error_str = str(e)
            print(f"Operation result: {error_str}")
            if "TS001" in error_str or "timeout" in error_str.lower():
                 print("Connection Timed Out - check URL.")
                 sys.exit(1)
            # If it says 'Invalid login credentials', that means it CONNECTED and validated the request!
    
    except Exception as e:
        print(f"Faile to load secrets or initialize: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify()
