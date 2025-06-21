import os
import base64
import mimetypes
import re
import io
import time
import uuid
import json
import threading
from flask import Flask, request, Response, render_template
import requests
from bs4 import BeautifulSoup
import torch
from diffusers import StableDiffusionPipeline
import diffusers

OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/generate')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral')

app = Flask(__name__)
pipe = None # Global pipeline object
tasks = {} # Dictionary to hold the state of background tasks

# --- Tame the logs ---
diffusers.logging.set_verbosity_error()

def load_sd_model():
    """Loads the Stable Diffusion model if it's not already loaded."""
    global pipe
    if pipe is None:
        print("Loading Stable Diffusion model...")
        # Note: Using float32 as float16 can be problematic on CPU
        # Disabling the safety checker to prevent low-level bus errors on some hardware.
        loaded_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float32,
            safety_checker=None
        )
        if torch.cuda.is_available():
            print("CUDA is available, moving model to GPU.")
            pipe = loaded_pipe.to("cuda")
        else:
            print("CUDA not available, using CPU.")
            pipe = loaded_pipe.to("cpu")
        print("Stable Diffusion model loaded.")

def get_asset_as_data_url(url):
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        content_type = response.headers.get('content-type', mimetypes.guess_type(url)[0] or 'application/octet-stream')
        encoded_content = base64.b64encode(response.content).decode('utf-8')
        return f"data:{content_type};base64,{encoded_content}"
    except Exception as e:
        print(f"[ERROR] Could not fetch asset {url}: {e}")
        return None

def inline_assets(html_content, generated_image_data_url=None):
    soup = BeautifulSoup(html_content, 'html.parser')
    if generated_image_data_url:
        img_tag = soup.find('img', id='generated-image')
        if img_tag:
            img_tag['src'] = generated_image_data_url

    for tag in soup.find_all(['img', 'script'], src=True):
        if tag.get('id') != 'generated-image':
            if tag['src'].startswith('http'):
                data_url = get_asset_as_data_url(tag['src'])
                if data_url:
                    tag['src'] = data_url
                else:
                    tag.decompose()
            else:
                tag.decompose()
    
    for tag in soup.find_all('link', href=True, rel='stylesheet'):
        if tag['href'].startswith('http'):
            data_url = get_asset_as_data_url(tag['href'])
            if data_url:
                tag['href'] = data_url
            else:
                tag.decompose()
        else:
            tag.decompose()
            
    return str(soup)

def build_prompt(context):
    prompt = f"""
You are a creative web page generator. Your only output should be a valid HTML document.
Generate a complete, single, visually appealing HTML page about a random, interesting topic.
- The page must include one and only one `<img>` tag with `id="generated-image"` and an empty `src` attribute.
- Immediately before the `<img>` tag, include an HTML comment with an image prompt like this: `<!-- image_prompt: a photorealistic image of a cat programming -->`
- DO NOT use any markdown.
- Your response must start with `<!DOCTYPE html>` and end with `</html>`.

The context for this generation is: {context}
"""
    return prompt

def query_ollama(prompt):
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        print(f"[DEBUG] Querying Ollama...")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        print(f"[DEBUG] Ollama query successful.")
        return response.json().get('response', ''), None
    except Exception as e:
        return None, f"Error querying Ollama: {e}"

def generate_image(prompt):
    if not prompt:
        return None, "No image prompt provided."

    load_sd_model() # Ensure the model is loaded before use

    try:
        # Sanitize the prompt to remove problematic characters and limit length
        clean_prompt = re.sub(r'[^a-zA-Z0-9\s,.]', '', prompt).strip()
        clean_prompt = clean_prompt[:200] # Truncate to a reasonable length

        print(f"[DEBUG] Generating image with sanitized prompt: {clean_prompt}")
        with torch.no_grad():
            image = pipe(clean_prompt, num_inference_steps=100).images[0]
        
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        print("[DEBUG] Image generated successfully.")
        return f"data:image/png;base64,{img_str}", None
    except Exception as e:
        print(f"[ERROR] Error generating image: {e}")
        return None, f"Error generating image: {e}"

def run_generation_task(task_id, context):
    try:
        start_time = time.time()
        print(f"\n--- [{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting generation task {task_id} ---")
        
        # Step 1: Query Ollama
        tasks[task_id].update({'status': 'Generating HTML with Ollama...', 'progress': 10})
        prompt = build_prompt(context)
        html, err = query_ollama(prompt)
        if err or not html:
            raise Exception(err or 'Ollama returned an empty response.')
        
        # Step 2: Extract image prompt
        tasks[task_id].update({'status': 'Extracting image prompt...', 'progress': 40})
        match = re.search(r'<!--\s*image_prompt:\s*(.*?)\s*-->', html)
        image_prompt = match.group(1) if match else "a random beautiful landscape"

        # Step 3: Generate Image
        tasks[task_id].update({'status': 'Generating image with Stable Diffusion...', 'progress': 50})
        generated_image_data_url, err = generate_image(image_prompt)
        if err:
            print(f"[WARNING] Task {task_id}: Could not generate image: {err}")
        
        # Step 4: Inline assets
        tasks[task_id].update({'status': 'Inlining assets and finishing up...', 'progress': 90})
        soup = BeautifulSoup(html, 'html.parser')
        for a_tag in soup.find_all('a'):
            a_tag['data-context'] = context
        final_html = inline_assets(str(soup), generated_image_data_url)

        # Step 5: Finalize
        tasks[task_id].update({'status': 'Complete', 'progress': 100, 'result': final_html})
        print(f"--- Task {task_id} finished in {time.time() - start_time:.2f}s ---")

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}")
        tasks[task_id].update({'status': f'Failed: {e}', 'progress': 100, 'result': f"<h1>Error</h1><p>The page generation failed: {e}</p>"})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def generate(path):
    task_id = str(uuid.uuid4())
    user_agent = request.headers.get('User-Agent', 'Unknown')
    ip_address = request.remote_addr
    context = f"Request for '{request.path}' from IP: {ip_address}, User-Agent: {user_agent}"
    
    tasks[task_id] = {'status': 'Pending', 'progress': 0}

    thread = threading.Thread(target=run_generation_task, args=(task_id, context))
    thread.start()

    return render_template('loading.html', task_id=task_id)

@app.route('/stream/<task_id>')
def stream(task_id):
    def event_stream():
        last_progress = -1
        while True:
            time.sleep(1)
            task = tasks.get(task_id)
            if not task:
                break 

            current_progress = task.get('progress', 0)
            if current_progress > last_progress:
                data = {'status': task['status'], 'progress': task['progress']}
                if task['progress'] >= 100:
                    data['html'] = task['result']
                
                yield f"data: {json.dumps(data)}\n\n"
                last_progress = current_progress

            if task['progress'] >= 100:
                break
    
    return Response(event_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Pre-load the model on startup to avoid a delay on the first request.
    load_sd_model()
    # Note: Flask's development server is not ideal for managing threads in production.
    # A proper WSGI server like Gunicorn or uWSGI should be used.
    # The `threaded=False` is kept to serialize the main requests, but generation is now in background threads.
    app.run(debug=True, use_reloader=False, port=5001, threaded=True) 