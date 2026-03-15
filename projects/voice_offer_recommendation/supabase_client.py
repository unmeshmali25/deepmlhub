from supabase import create_client
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Validate environment variables are set
if not SUPABASE_URL:
    raise ValueError(
        "SUPABASE_URL environment variable is not set. Please check your .env file."
    )
if not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_KEY environment variable is not set. Please check your .env file."
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
