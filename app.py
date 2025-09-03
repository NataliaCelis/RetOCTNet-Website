import gradio as gr
import numpy as np
from usage_utils import drn_execute
from PIL import Image

def segment_oct(image):
    # Save uploaded image
    image.save("temp.tiff")

    # Run inference
    images, predictions = drn_execute(["temp.tiff"], discard=False)

    # Convert prediction to uint8
    output = Image.fromarray(np.squeeze(predictions).astype(np.uint8))
    return output

iface = gr.Interface(
    fn=segment_oct,
    inputs=gr.Image(type="pil", label="Upload OCT Image"),
    outputs=gr.Image(type="pil", label="Segmentation Output"),
    title="OCT Image Segmentation",
    description="Upload an OCT scan (TIFF/PNG/JPEG) and get segmentation results."
)

if __name__ == "__main__":
    iface.launch()
