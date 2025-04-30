# server.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import outetts
import os
import json

app = Flask(__name__)
CORS(app)

# Static folder to serve audio files
OUTPUT_FOLDER = 'static'
AUDIO_FILE = 'output.wav'

# Ensure the static folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize the outetts interface
interface = outetts.Interface(
    config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,
        backend=outetts.Backend.LLAMACPP,
        quantization=outetts.LlamaCppQuantization.FP16
    )
)
speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")


@app.route('/talk', methods=['POST'])
def talk():
    data = request.get_json()
    text = data.get("text", "")

    # Generate audio using outetts
    output = interface.generate(
        config=outetts.GenerationConfig(
            text=text,
            generation_type=outetts.GenerationType.CHUNKED,
            speaker=speaker,
            sampler_config=outetts.SamplerConfig(temperature=0.4)
        )
    )
    output_path = os.path.join(OUTPUT_FOLDER, AUDIO_FILE)
    output.save(output_path)

    # Simulate or load actual blendData from some real logic
    with open("blendDataBlink.json", "r") as f:
        dummy_blend_data = json.load(f)  # Replace this with real sync logic later

    return jsonify({
        "filename": f"/static/{AUDIO_FILE}",
        "blendData": dummy_blend_data
    })


@app.route('/static/<path:filename>')
def serve_audio(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
