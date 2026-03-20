import sys
from pathlib import Path

# Make the repo parent visible so `import pse` resolves to this repo root.
# In CI: /home/runner/work/pse/pse/ → parent /home/runner/work/pse/ → pse/ found.
# Locally: c:\dev\northflow\pse\ → parent c:\dev\northflow\ → pse\ found.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

# Load .env so ERA5_CDS_API_KEY and other secrets are available to all tests.
load_dotenv()
