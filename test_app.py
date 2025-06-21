import pytest
from unittest.mock import patch, MagicMock
from app import app as flask_app

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # Prevent the real model from loading during tests
    with patch('app.load_sd_model') as mock_load_model:
        yield flask_app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

def test_generate_endpoint_integration(client):
    """
    Test the main endpoint in an integrated way, mocking only the AI/ML functions.
    This test verifies that the Flask routing, request handling, and HTML processing work.
    """
    # Mock the return value of the Ollama call
    mock_ollama_response = (
        '<!DOCTYPE html><html><body><h1 id="test">Test Page</h1><!-- image_prompt: a test image --> <img id="generated-image" src=""></body></html>',
        None
    )
    
    # Mock the return value of the image generation call
    mock_image_response = ('data:image/png;base64,fakeimagedata', None)

    with patch('app.query_ollama', return_value=mock_ollama_response) as mock_ollama:
        with patch('app.generate_image', return_value=mock_image_response) as mock_image_gen:
            # Make a request to the app
            response = client.get('/test-path')

            # Assert that the app returned a successful response
            assert response.status_code == 200
            assert response.mimetype == 'text/html'
            
            # Assert that our mocked functions were called
            mock_ollama.assert_called_once()
            mock_image_gen.assert_called_once_with('a test image')

            # Assert that the final HTML contains the mocked data
            assert b'Test Page' in response.data
            assert b'data:image/png;base64,fakeimagedata' in response.data
            assert b'data-context' in response.data 