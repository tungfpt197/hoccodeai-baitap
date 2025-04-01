import torch
from PIL.Image import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(raw_image: Image) -> str:
    inputs = processor(raw_image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption
    
# Input đầu vào là một ảnh
image = gr.Image(label="Ảnh đầu vào")

# Output là đầu ra là chú thích, một string về ảnh
caption = gr.Textbox(label="Chú thích")

# Dựng interface với hàm generate_caption đã viết, nhận input là ảnh và hiển thị chú thích
gr.Interface(
    fn=generate_caption,
    inputs=image,
    outputs=caption,
    title="Nhận diện ảnh cùng BLIP").launch()
    
