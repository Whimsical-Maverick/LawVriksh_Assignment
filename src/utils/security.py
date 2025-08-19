from fastapi import Header, HTTPException, status, Depends
from typing import Optional
import os

# Load API key from environment
API_KEY = os.getenv("API_KEY")

def api_key_required(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Dependency function to enforce API key authentication.
    Looks for the API key in the 'X-API-Key' request header.
    """
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return x_api_key