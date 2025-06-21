# Dynamic Web Page Generator with Stable Diffusion

This Flask application dynamically generates a unique, single-page HTML website for any requested path. It uses the Ollama service with the Mistral model to generate the HTML structure and content, and leverages a local Stable Diffusion model (`runwayml/stable-diffusion-v1-5`) via the `diffusers` library to create and inline a relevant image.

## Features

-   **Catch-all Routing**: Responds to a request for any URL path.
-   **AI-Generated Content**: Uses Ollama/Mistral to create HTML based on request context (IP, User-Agent).
-   **Local Image Generation**: Creates a unique image for each page using Stable Diffusion.
-   **Fully Inlined Assets**: All CSS, JS, and images are embedded directly into the HTML as data URLs for a dependency-free, single-file output.
-   **Context-Aware Links**: Injects request context into `data-context` attributes on all `<a>` tags.

## Setup

This project uses `torch` and `diffusers`, which can be large. The setup script will create a Python virtual environment to manage dependencies.

1.  **Run the setup script:**
    
    For macOS and Linux:
    ```bash
    ./setup.sh
    ```

2.  **Ensure Ollama is running** and the `mistral` model is available. The setup script will remind you, but you can pull the model with:
    ```bash
    ollama pull mistral
    ```

## Running the Application

Once the setup is complete, run the Flask server:

```bash
python app.py
```

The server will start on `http://localhost:5001`. The first time you run it, the Stable Diffusion model will be downloaded (several gigabytes), which may take some time. Subsequent runs will use the cached model.

The application runs in a single-threaded mode to handle resource-intensive image generation requests sequentially.

---

**Note:** This project assumes Ollama is running locally and accessible via its default API endpoint. 