import os
import base64
import mimetypes
import re
import io
import time
import uuid
import json
import threading
from functools import partial
from flask import Flask, request, Response, render_template
import requests
from bs4 import BeautifulSoup
import torch
from diffusers import StableDiffusionPipeline
from diffusers.callbacks import PipelineCallback
import diffusers

# Use 127.0.0.1 instead of 127.0.0.1 for better deployment compatibility
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://127.0.0.1:11434/api/generate')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral')

app = Flask(__name__)
pipe = None # Global pipeline object
tasks = {} # Dictionary to hold the state of background tasks
active_requests = set() # Track active requests to prevent duplicates

# --- Tame the logs ---
diffusers.logging.set_verbosity_error()

class ProgressCallback(PipelineCallback):
    def __init__(self, task_id):
        super().__init__()
        self.task_id = task_id
        
    @property
    def tensor_inputs(self):
        return ["latents"]
        
    def callback_fn(self, pipeline, step_index, timesteps, callback_kwargs):
        try:
            task = tasks.get(self.task_id)
            if not task or task.get('should_stop'):
                if task:
                    task['status'] = "Cancelled by user."
                raise Exception("Task cancelled by user.")

            latents = callback_kwargs.get("latents")
            if latents is None:
                return callback_kwargs

            num_steps = task.get('num_inference_steps', 100)
            if num_steps is None:
                num_steps = 100
                
            step = step_index
            progress = 50 + int(((step + 1) / num_steps) * 40) # Use step + 1 for user-facing count
            task.update({
                'status': f'Generating image... Step {step + 1}/{num_steps}',
                'progress': progress
            })

            if (step + 1) > 0 and (step + 1) % 10 == 0:
                try:
                    latents_scaled = 1 / pipeline.vae.config.scaling_factor * latents
                    with torch.no_grad():
                        image_tensor = pipeline.vae.decode(latents_scaled).sample
                    image = pipeline.image_processor.postprocess(image_tensor, output_type='pil')[0]
                    
                    thumbnail = image.copy()
                    thumbnail.thumbnail((128, 128))
                    
                    buf = io.BytesIO()
                    thumbnail.save(buf, format="JPEG", quality=75)
                    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                    
                    task.update({'thumbnail': f"data:image/jpeg;base64,{img_str}"})
                    print(f"[DEBUG] Task {self.task_id}: Sent thumbnail at step {step + 1}")
                except Exception as e:
                    print(f"[ERROR] Task {self.task_id}: Could not generate thumbnail at step {step + 1}: {e}")
            
            return callback_kwargs
        except Exception as e:
            print(f"[ERROR] Task {self.task_id}: Callback error: {e}")
            return callback_kwargs

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

REQUIREMENTS:
- The page MUST include CSS animations (keyframes, transitions, or transforms)
- The page MUST include JavaScript functionality for dynamic or interactive behaviors (event-driven updates, client-side logic, DOM manipulation, etc.)
- The page must include one and only one `<img>` tag with `id="generated-image"` and an empty `src` attribute.
- Immediately before the `<img>` tag, include an HTML comment with an image prompt like this: `<!-- image_prompt: a photorealistic image of a cat programming -->`
- DO NOT use any markdown.
- Your response must start with `<!DOCTYPE html>` and end with `</html>`.

EXAMPLES OF REQUIRED FEATURES:
- CSS animations: hover effects, loading animations, color transitions, movement effects
- JavaScript: interactive buttons, form validation, dynamic content updates, event listeners, DOM manipulation

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

def generate_image(prompt, task_id, num_inference_steps=100):
    if not prompt:
        return None, "No image prompt provided."

    load_sd_model()

    try:
        clean_prompt = re.sub(r'[^a-zA-Z0-9\s,.]', '', prompt).strip()
        clean_prompt = clean_prompt[:200]

        print(f"[DEBUG] Generating image with sanitized prompt: {clean_prompt}")
        
        # Use the class-based callback
        callback = ProgressCallback(task_id)
        
        with torch.no_grad():
            image = pipe(
                clean_prompt, 
                num_inference_steps=num_inference_steps,
                callback_on_step_end=callback
            ).images[0]
        
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        print("[DEBUG] Image generated successfully.")
        return f"data:image/png;base64,{img_str}", None
    except Exception as e:
        print(f"[ERROR] Error in generate_image: {e}")
        return None, f"Error generating image: {e}"

def create_fallback_html(context):
    """Create fallback HTML content when Ollama is not available."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dynamic Web Page Generator</title>
    <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }}
        
        .container {{
            max-width: 800px;
            margin: 0 auto;
            animation: fadeIn 1s ease-out;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .content {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        .feature-card {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        
        .feature-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            animation: pulse 2s infinite;
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ff6b6b;
            margin-right: 10px;
            animation: pulse 2s infinite;
        }}
        
        .button {{
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
            margin: 10px;
        }}
        
        .button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
        
        .generated-image {{
            width: 100%;
            max-width: 400px;
            height: 300px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 20px auto;
            color: white;
            font-size: 18px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Dynamic Web Page Generator</h1>
            <p>AI-Powered Content Creation Platform</p>
        </div>
        
        <div class="content">
            <h2><span class="status-indicator"></span>Service Status</h2>
            <p>Currently running in fallback mode. The AI content generation service (Ollama) is not available, but the application is fully functional for demonstration purposes.</p>
            
            <div class="feature-card">
                <h3>✨ Features</h3>
                <ul>
                    <li>Dynamic HTML Generation</li>
                    <li>CSS Animations & Transitions</li>
                    <li>Interactive JavaScript</li>
                    <li>Real-time Image Generation</li>
                    <li>Context-Aware Content</li>
                    <li>Fully Inlined Assets</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h3>🔧 Technical Details</h3>
                <p><strong>Request Context:</strong> {context}</p>
                <p><strong>Ollama Status:</strong> <span style="color: #ff6b6b;">Not Available</span></p>
                <p><strong>Image Generation:</strong> <span style="color: #00b894;">Available (Stable Diffusion)</span></p>
            </div>
        </div>
        
        <div class="content">
            <h2>🎨 Generated Content Preview</h2>
            <div class="generated-image">
                <div>
                    <p>📸 Image Generation Available</p>
                    <p><small>Stable Diffusion model loaded and ready</small></p>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <button class="button" onclick="showFeatures()">Show Features</button>
                <button class="button" onclick="animateContent()">Animate Content</button>
                <button class="button" onclick="generateRandomContent()">Generate Random</button>
            </div>
        </div>
    </div>
    
    <!-- image_prompt: a modern web development workspace with multiple monitors showing code and design tools -->
    <img id="generated-image" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" style="display: none;">
    
    <script>
        // JavaScript functionality for interactive features
        let animationCount = 0;
        
        function showFeatures() {{
            const features = [
                'Dynamic HTML Generation',
                'CSS Animations & Transitions',
                'Interactive JavaScript',
                'Real-time Image Generation',
                'Context-Aware Content',
                'Fully Inlined Assets'
            ];
            
            const content = document.querySelector('.content');
            const featureList = document.createElement('div');
            featureList.className = 'feature-card';
            featureList.innerHTML = '<h3>🎯 Active Features</h3><ul>' + 
                features.map(f => '<li>' + f + '</li>').join('') + '</ul>';
            
            content.appendChild(featureList);
            featureList.style.animation = 'fadeIn 0.5s ease-out';
        }}
        
        function animateContent() {{
            animationCount++;
            const cards = document.querySelectorAll('.feature-card');
            cards.forEach((card, index) => {{
                setTimeout(() => {{
                    card.style.transform = 'rotate(5deg) scale(1.05)';
                    setTimeout(() => {{
                        card.style.transform = 'rotate(0deg) scale(1)';
                    }}, 300);
                }}, index * 100);
            }});
        }}
        
        function generateRandomContent() {{
            const topics = [
                'Artificial Intelligence',
                'Web Development',
                'Data Science',
                'Machine Learning',
                'Cloud Computing',
                'Cybersecurity'
            ];
            
            const randomTopic = topics[Math.floor(Math.random() * topics.length)];
            const header = document.querySelector('.header h1');
            header.textContent = '🚀 ' + randomTopic + ' Hub';
            header.style.animation = 'pulse 1s ease-in-out';
        }}
        
        // Add event listeners for enhanced interactivity
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('Dynamic Web Page Generator loaded successfully!');
            
            // Add hover effects to all interactive elements
            const buttons = document.querySelectorAll('.button');
            buttons.forEach(button => {{
                button.addEventListener('mouseenter', function() {{
                    this.style.background = 'linear-gradient(45deg, #00b894, #00a085)';
                }});
                button.addEventListener('mouseleave', function() {{
                    this.style.background = 'linear-gradient(45deg, #ff6b6b, #ee5a24)';
                }});
            }});
            
            // Add click effects to feature cards
            const cards = document.querySelectorAll('.feature-card');
            cards.forEach(card => {{
                card.addEventListener('click', function() {{
                    this.style.background = 'rgba(255, 255, 255, 0.25)';
                    setTimeout(() => {{
                        this.style.background = 'rgba(255, 255, 255, 0.15)';
                    }}, 200);
                }});
            }});
        }});
    </script>
</body>
</html>"""

def run_generation_task(task_id, context):
    try:
        start_time = time.time()
        print(f"\n--- [{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting generation task {task_id} ---")
        
        tasks[task_id].update({'status': 'Generating HTML with Ollama...', 'progress': 10})
        prompt = build_prompt(context)
        html, err = query_ollama(prompt)
        
        # If Ollama is not available, use fallback HTML
        if err or not html:
            if "Ollama service not available" in str(err) or "404" in str(err):
                print(f"[INFO] Task {task_id}: Using fallback HTML due to Ollama unavailability")
                html = create_fallback_html(context)
            else:
                raise Exception(err or 'Ollama returned an empty response.')
        
        tasks[task_id].update({'status': 'Extracting image prompt...', 'progress': 40})
        match = re.search(r'<!--\s*image_prompt:\s*(.*?)\s*-->', html)
        image_prompt = match.group(1) if match else "a random beautiful landscape"

        num_inference_steps = 100
        tasks[task_id].update({
            'status': 'Generating image with Stable Diffusion...', 
            'progress': 50,
            'num_inference_steps': num_inference_steps
        })
        generated_image_data_url, err = generate_image(image_prompt, task_id, num_inference_steps=num_inference_steps)
        if err:
            print(f"[WARNING] Task {task_id}: Could not generate image: {err}")
            generated_image_data_url = None
        
        tasks[task_id].update({'status': 'Inlining assets and finishing up...', 'progress': 95})
        soup = BeautifulSoup(html, 'html.parser')
        for a_tag in soup.find_all('a'):
            a_tag['data-context'] = context
        final_html = inline_assets(str(soup), generated_image_data_url)

        tasks[task_id].update({'status': 'Complete', 'progress': 100, 'result': final_html})
        print(f"--- Task {task_id} finished in {time.time() - start_time:.2f}s ---")

    except Exception as e:
        if "Task cancelled by user" in str(e):
            print(f"--- Task {task_id} gracefully stopped by user. ---")
        else:
            print(f"[ERROR] Task {task_id} failed: {e}")
            if task_id in tasks:
                 tasks[task_id].update({'status': f'Failed: {e}', 'progress': 100, 'result': f"<h1>Error</h1><p>The page generation failed: {e}</p>"})
    finally:
        # Add a delay to ensure the client receives the final result before cleanup
        if task_id in tasks:
            print(f"[DEBUG] Task {task_id} completed, waiting 5 seconds before cleanup...")
            time.sleep(5)  # Give client time to receive final result
            if task_id in tasks:
                del tasks[task_id]
                print(f"[DEBUG] Task {task_id} cleaned up.")

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def generate(path):
    # Create a unique request identifier to prevent duplicates
    request_id = f"{request.remote_addr}_{request.headers.get('User-Agent', 'Unknown')}_{path}"
    
    # Check if this request is already being processed
    if request_id in active_requests:
        print(f"[INFO] Duplicate request detected for {request_id}, returning existing task")
        # Find the existing task for this request
        for task_id, task in tasks.items():
            if task.get('request_id') == request_id:
                return render_template('loading.html', task_id=task_id)
    
    task_id = str(uuid.uuid4())
    user_agent = request.headers.get('User-Agent', 'Unknown')
    ip_address = request.remote_addr
    context = f"Request for '{request.path}' from IP: {ip_address}, User-Agent: {user_agent}"
    
    # Mark this request as active
    active_requests.add(request_id)
    tasks[task_id] = {
        'status': 'Pending', 
        'progress': 0,
        'request_id': request_id
    }

    thread = threading.Thread(target=run_generation_task, args=(task_id, context))
    thread.start()

    return render_template('loading.html', task_id=task_id)

@app.route('/stream/<task_id>')
def stream(task_id):
    # Check if task exists before starting the stream
    if task_id not in tasks:
        return Response("Task not found", status=404, mimetype='text/plain')
    
    def event_stream():
        last_progress = -1
        last_thumbnail = None
        last_status = None
        final_result_sent = False
        
        try:
            while True:
                time.sleep(0.5)
                task = tasks.get(task_id)
                if not task:
                    print(f"[INFO] Task {task_id} not found, ending stream")
                    break 

                current_progress = task.get('progress', 0)
                current_thumbnail = task.get('thumbnail', None)
                current_status = task.get('status', '')

                # Check if anything has changed
                has_changes = (current_progress > last_progress or 
                             current_thumbnail != last_thumbnail or 
                             current_status != last_status)

                if has_changes:
                    data = {'status': current_status, 'progress': current_progress}
                    if current_thumbnail:
                        data['thumbnail'] = current_thumbnail
                    if current_progress >= 100 and not final_result_sent:
                        data['html'] = task['result']
                        final_result_sent = True
                        print(f"[DEBUG] Task {task_id}: Sent final result to client")
                    
                    yield f"data: {json.dumps(data)}\n\n"
                    last_progress = current_progress
                    last_thumbnail = current_thumbnail
                    last_status = current_status

                # Stop conditions - only when task is complete or cancelled
                if current_progress >= 100 or task.get('should_stop'):
                    if final_result_sent:
                        print(f"[DEBUG] Task {task_id}: Final result sent, ending stream")
                        break
                    else:
                        # Wait a bit more to ensure final result is sent
                        time.sleep(1)
                    
        except GeneratorExit:
            print(f"[INFO] Client disconnected for task {task_id}. Flagging for cancellation.")
            if task_id in tasks:
                tasks[task_id]['should_stop'] = True
        except Exception as e:
            print(f"[ERROR] Stream error for task {task_id}: {e}")
        finally:
            print(f"[INFO] Stream closed for task {task_id}.")
            # Clean up the request tracking when the stream ends
            if task_id in tasks:
                request_id = tasks[task_id].get('request_id')
                if request_id in active_requests:
                    active_requests.remove(request_id)

    return Response(event_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    load_sd_model()
    app.run(debug=True, use_reloader=False, port=5001, threaded=True) 