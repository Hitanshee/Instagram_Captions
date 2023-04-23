from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import base64

app = Flask(__name__)
generator = None

def load_model():
    global generator
    global processor, model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    generator = pipeline('text-generation', model='gpt2')

@app.before_first_request
def before_first_request():
    load_model()


@app.route('/')
def index():
    return render_template('input.html')

def caption_gen(img_path):
    raw_image = Image.open(img_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    return str(processor.decode(out[0], skip_special_tokens=True))

@app.route('/generate', methods=['POST'])

def generate_captions():
    image_path = request.files['image_file']
    file_bytes = image_path.read()
    file_base64 = base64.b64encode(file_bytes).decode('ascii')
    image_description = caption_gen(image_path)
    num_captions = int(request.form['num_captions'])
    captions = []
    for i in range(num_captions):
        caption = generator(f"{image_description}\n", max_length=50, do_sample=True, temperature=0.7)[0]['generated_text']
        #captions.append(caption)
        captions.append(caption[len(image_description)+1:])
    return render_template('output.html', image_base64=file_base64, output=image_description,  captions=captions)

if __name__ == '__main__':
    app.run(debug=True)
