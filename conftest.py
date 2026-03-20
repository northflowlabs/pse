from dotenv import load_dotenv

# Load .env from repo root so ERA5_CDS_API_KEY and other secrets are
# available to all tests without needing to export them manually.
load_dotenv()
