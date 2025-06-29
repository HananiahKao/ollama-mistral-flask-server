import pytest
from unittest.mock import patch, MagicMock
from app import app as flask_app
import os
import json
import base64

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

def test_catch_all_routing(client):
    """Test that the app responds to any URL path."""
    paths = ['/', '/random', '/deep/nested/path', '/with-query?param=value']
    
    with patch('app.query_ollama', return_value=('<!DOCTYPE html><html><body>Test</body></html>', None)):
        with patch('app.generate_image', return_value=('data:image/png;base64,test', None)):
            for path in paths:
                response = client.get(path)
                assert response.status_code == 200
                assert response.mimetype == 'text/html'

def test_html_generation_requirements(client):
    """Test that generated HTML meets the specified requirements."""
    mock_html = '''<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
    <style>
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .animated { animation: fadeIn 1s; }
    </style>
    <div class="animated">Content</div>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            console.log('JavaScript loaded');
        });
    </script>
    <!-- image_prompt: a beautiful landscape -->
    <img id="generated-image" src="">
</body>
</html>'''
    
    with patch('app.query_ollama', return_value=(mock_html, None)):
        with patch('app.generate_image', return_value=('data:image/png;base64,test', None)):
            response = client.get('/test')
            html_content = response.data.decode('utf-8')
            
            # Check for required elements
            assert 'data:image/png;base64,test' in html_content
            assert 'data-context' in html_content
            assert '@keyframes' in html_content  # CSS animations
            assert 'addEventListener' in html_content  # JavaScript functionality
            assert '<!-- image_prompt:' in html_content

def test_context_injection(client):
    """Test that request context is properly injected into links."""
    mock_html = '''<!DOCTYPE html>
<html><body>
    <a href="/link1">Link 1</a>
    <a href="/link2">Link 2</a>
    <!-- image_prompt: test image -->
    <img id="generated-image" src="">
</body></html>'''
    
    with patch('app.query_ollama', return_value=(mock_html, None)):
        with patch('app.generate_image', return_value=('data:image/png;base64,test', None)):
            response = client.get('/test', headers={'User-Agent': 'TestBrowser'})
            html_content = response.data.decode('utf-8')
            
            # Check that links have data-context attributes
            assert 'data-context' in html_content
            # Check that the context contains request information
            assert 'TestBrowser' in html_content

def test_image_generation_integration(client):
    """Test the complete image generation workflow."""
    mock_html = '''<!DOCTYPE html><html><body>
    <!-- image_prompt: a red car on a mountain road -->
    <img id="generated-image" src="">
    </body></html>'''
    
    with patch('app.query_ollama', return_value=(mock_html, None)):
        with patch('app.generate_image') as mock_gen:
            mock_gen.return_value = ('data:image/jpeg;base64,generated_image_data', None)
            
            response = client.get('/test')
            
            # Verify image generation was called with correct prompt
            mock_gen.assert_called_once_with('a red car on a mountain road')
            
            # Verify the generated image is in the response
            assert b'data:image/jpeg;base64,generated_image_data' in response.data

def test_error_handling(client):
    """Test error handling for various failure scenarios."""
    # Test Ollama API failure
    with patch('app.query_ollama', return_value=(None, 'Ollama API error')):
        response = client.get('/test')
        assert response.status_code == 500
    
    # Test image generation failure
    mock_html = '''<!DOCTYPE html><html><body>
    <!-- image_prompt: test -->
    <img id="generated-image" src="">
    </body></html>'''
    
    with patch('app.query_ollama', return_value=(mock_html, None)):
        with patch('app.generate_image', return_value=(None, 'Image generation failed')):
            response = client.get('/test')
            # Should still return HTML even if image generation fails
            assert response.status_code == 200
            assert b'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==' in response.data

def test_asset_inlining(client):
    """Test that external assets are properly inlined."""
    mock_html = '''<!DOCTYPE html><html><body>
    <img src="https://example.com/image.jpg">
    <link rel="stylesheet" href="https://example.com/style.css">
    <script src="https://example.com/script.js"></script>
    <!-- image_prompt: test -->
    <img id="generated-image" src="">
    </body></html>'''
    
    with patch('app.query_ollama', return_value=(mock_html, None)):
        with patch('app.generate_image', return_value=('data:image/png;base64,test', None)):
            with patch('app.get_asset_as_data_url') as mock_get_asset:
                mock_get_asset.return_value = 'data:text/plain;base64,dGVzdA=='
                
                response = client.get('/test')
                html_content = response.data.decode('utf-8')
                
                # Check that external assets were processed
                assert mock_get_asset.call_count >= 3  # At least 3 external assets

def test_streaming_endpoint(client):
    """Test the streaming endpoint for real-time updates."""
    with patch('app.tasks', {'test-task-id': {'status': 'Processing', 'progress': 25}}):
        response = client.get('/stream/test-task-id')
        assert response.status_code == 200
        assert response.mimetype == 'text/event-stream'

def test_deployment_environment_simulation(client):
    """Simulate deployed environment conditions."""
    # Test with deployment-like environment variables
    original_env = os.environ.copy()
    os.environ['PORT'] = '8080'
    os.environ['OLLAMA_API_URL'] = 'http://127.0.0.1:11434/api/generate'
    os.environ['OLLAMA_MODEL'] = 'mistral'
    
    try:
        with patch('app.query_ollama', return_value=('<!DOCTYPE html><html><body>Test</body></html>', None)):
            with patch('app.generate_image', return_value=('data:image/png;base64,test', None)):
                response = client.get('/deployment-test')
                assert response.status_code == 200
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

def test_performance_characteristics(client):
    """Test performance characteristics and resource usage."""
    import time
    
    mock_html = '''<!DOCTYPE html><html><body>
    <!-- image_prompt: performance test -->
    <img id="generated-image" src="">
    </body></html>'''
    
    with patch('app.query_ollama', return_value=(mock_html, None)):
        with patch('app.generate_image', return_value=('data:image/png;base64,test', None)):
            start_time = time.time()
            response = client.get('/performance-test')
            end_time = time.time()
            
            # Response should complete within reasonable time (5 seconds)
            assert end_time - start_time < 5.0
            assert response.status_code == 200

def test_concurrent_requests(client):
    """Test handling of concurrent requests."""
    import threading
    import time
    
    mock_html = '''<!DOCTYPE html><html><body>
    <!-- image_prompt: concurrent test -->
    <img id="generated-image" src="">
    </body></html>'''
    
    responses = []
    errors = []
    
    def make_request():
        try:
            with patch('app.query_ollama', return_value=(mock_html, None)):
                with patch('app.generate_image', return_value=('data:image/png;base64,test', None)):
                    response = client.get('/concurrent-test')
                    responses.append(response.status_code)
        except Exception as e:
            errors.append(str(e))
    
    # Start multiple concurrent requests
    threads = []
    for i in range(3):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # All requests should succeed
    assert len(errors) == 0
    assert all(status == 200 for status in responses)

def test_memory_usage():
    """Test that the app doesn't have memory leaks."""
    import psutil
    import gc
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Make multiple requests to simulate usage
    with patch('app.query_ollama', return_value=('<!DOCTYPE html><html><body>Test</body></html>', None)):
        with patch('app.generate_image', return_value=('data:image/png;base64,test', None)):
            for i in range(10):
                with flask_app.test_client() as client:
                    client.get(f'/memory-test-{i}')
    
    # Force garbage collection
    gc.collect()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (less than 100MB)
    assert memory_increase < 100 * 1024 * 1024

def test_fallback_functionality(client):
    """Test that the app works with fallback HTML when Ollama is not available."""
    # Test with Ollama unavailable
    with patch('app.query_ollama', return_value=(None, "Ollama service not available")):
        with patch('app.generate_image', return_value=('data:image/png;base64,test', None)):
            response = client.get('/fallback-test')
            assert response.status_code == 200
            html_content = response.data.decode('utf-8')
            
            # Check for fallback HTML content
            assert 'Dynamic Web Page Generator' in html_content
            assert 'fallback mode' in html_content
            assert 'Ollama Status' in html_content
            assert 'Not Available' in html_content
            assert 'data-context' in html_content
            assert '@keyframes' in html_content  # CSS animations
            assert 'addEventListener' in html_content  # JavaScript functionality

def test_ollama_url_fallback(client):
    """Test that the app tries multiple Ollama URLs and falls back gracefully."""
    # Mock the check_ollama_availability function
    with patch('app.check_ollama_availability', return_value=None):
        with patch('app.generate_fallback_html') as mock_fallback:
            mock_fallback.return_value = '<!DOCTYPE html><html><body>Fallback</body></html>'
            
            response = client.get('/url-fallback-test')
            assert response.status_code == 200

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 