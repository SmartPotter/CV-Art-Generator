import gradio as gr
from PIL import Image
from style_transfer import run_style_transfer
import tempfile
import os

STYLE_CHOICES = {
    "Starry Night": "assets/styles/starry-night.jpg",
    "Rain Princess": "assets/styles/rain-princess.jpg",
    "Candy": "assets/styles/candy.jpg",
    "Mosaic": "assets/styles/mosaic.jpg",
    "Upload your own": None
}

def transfer_style(content_img, style_choice, uploaded_style_img):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as cf:
        content_path = cf.name
        content_img.save(content_path)

    if STYLE_CHOICES[style_choice] is not None:
        style_path = STYLE_CHOICES[style_choice]
        style_img = Image.open(style_path)
    else:
        if uploaded_style_img is None:
            os.remove(content_path)
            raise gr.Error("Please upload a style image or select a predefined style.")
        style_img = uploaded_style_img

    if STYLE_CHOICES[style_choice] is not None:
        pass
    else:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as sf:
            style_path = sf.name
            style_img.save(style_path)

    # Run style transfer
    result = run_style_transfer(content_path, style_path)

    os.remove(content_path)
    if STYLE_CHOICES[style_choice] is None:
        os.remove(style_path)

    return Image.fromarray(result)

with gr.Blocks() as iface:
    gr.Markdown("# Neural Style Transfer")
    gr.Markdown("Upload a content image and pick a famous style or upload your own style image.")

    with gr.Row():
        content_input = gr.Image(type="pil", label="Content Image")
        with gr.Column():
            style_choice = gr.Dropdown(
                choices=list(STYLE_CHOICES.keys()),
                value="Starry Night",
                label="Choose a Style"
            )
            uploaded_style_img = gr.Image(type="pil", label="Upload Style Image (if selected)", visible=False)

    def toggle_upload(choice):
        return gr.update(visible=(choice == "Upload your own"))

    style_choice.change(toggle_upload, style_choice, uploaded_style_img)

    output = gr.Image(label="Stylized Output")

    btn = gr.Button("Stylize")
    btn.click(
        transfer_style,
        inputs=[content_input, style_choice, uploaded_style_img],
        outputs=output
    )

if __name__ == "__main__":
    iface.launch()
