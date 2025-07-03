# Neural Style Transfer App

A creative AI tool that lets you **blend iconic art styles into real-world photos** using deep learning. Powered by VGG19 and TensorFlow, this interactive **Gradio app** supports **multiple style images**, **custom weights**, and **on-the-fly rendering** of artistic outputs.


## Features

- Upload your own content image
- Blend **multiple style images** with adjustable weights *(first one used currently)*
- Control the number of training iterations
- Built using **VGG19** pretrained on ImageNet
- Download your stylized result
- Deployable via Hugging Face Spaces or locally via Gradio


## Example Output

| Content Image | Style Image(s) | Stylized Output |
|---------------|----------------|------------------|
| ![content](images/city.png) | ![style](images/starry.jpg) | ![output](images/paris_generated_at_iteration_2000.png) |

> Example: St. Basil’s Cathedral styled with *Starry Night* by Vincent van Gogh

---

## Quick Start

### Setup Locally

```bash
git clone https://github.com/SmartPotter/CV-Art-Generator.git
cd CV-Art-Generator

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python app_gradio.py