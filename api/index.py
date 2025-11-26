from fastapi import FastAPI
from mangum import Mangum  # For ASGI compatibility with serverless

from deploy import app  # Import your FastAPI app

handler = Mangum(app)