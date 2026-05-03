# app.py - Entry point for production deployment
# Direct import of the FastAPI app

from api.main import app

print(f"App loaded: {app}")
print(f"App routes: {len(app.routes)}")

