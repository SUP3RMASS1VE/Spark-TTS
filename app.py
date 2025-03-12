import os
import torch
import soundfile as sf
import logging
import argparse
import gradio as gr
from datetime import datetime
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI
from huggingface_hub import snapshot_download
from tqdm import tqdm

MODEL = None

def initialize_model(model_dir=None, device="cpu"):
    """Load the model once at the beginning."""
    
    if model_dir is None:
        logging.info(f"Downloading model to: {model_dir}")
        print("Downloading model from HuggingFace Hub...")
        model_dir = snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir="pretrained_models/Spark-TTS-0.5B")
        
    logging.info(f"Loading model from: {model_dir}")
    print(f"Loading model from: {model_dir}")
    
    # Create a progress bar for model loading
    with tqdm(total=3, desc="Loading model components", ncols=100) as pbar:
        pbar.set_description("Setting up device")
        device = torch.device(device)
        pbar.update(1)
        
        pbar.set_description("Initializing SparkTTS")
        model = SparkTTS(model_dir, device)
        pbar.update(1)
        
        pbar.set_description("Model loaded successfully")
        pbar.update(1)
    
    print(f"Model loaded successfully on {device}")
    return model

def generate(text,
            prompt_speech,
            prompt_text,
            gender,
            pitch,
            speed,
):
    """Generate audio from text."""
    
    global MODEL
    
    # Initialize model if not already done
    if MODEL is None:
        print("Initializing model...")
        MODEL = initialize_model(device="cuda" if torch.cuda.is_available() else "cpu")
    
    model = MODEL
    
    # if gpu available, move model to gpu
    if torch.cuda.is_available():
        print("Moving model to GPU")
        model.to("cuda")
    
    print(f"Generating speech for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # Create a wrapper for the inference method to add progress tracking
    original_generate = model.model.generate
    
    def generate_with_progress(*args, **kwargs):
        # Extract max_new_tokens from kwargs
        max_new_tokens = kwargs.get('max_new_tokens', 3000)
        
        # Create a progress bar
        progress_bar = tqdm(total=100, desc="Generating speech", ncols=100)
        
        # Store the original forward method
        original_forward = model.model.forward
        
        # Define a new forward method that updates the progress bar
        def forward_with_progress(*fargs, **fkwargs):
            # Call the original forward method
            result = original_forward(*fargs, **fkwargs)
            
            # Update the progress bar based on the current generation step
            # This is an approximation since we don't have direct access to the generation step
            if hasattr(result, 'logits') and hasattr(model.model, 'generation_config'):
                # Estimate progress based on output sequence length
                if fargs and isinstance(fargs[0], torch.Tensor):
                    current_length = fargs[0].size(1)
                    # Estimate progress percentage
                    progress = min(int((current_length / (current_length + max_new_tokens)) * 100), 99)
                    progress_bar.update(progress - progress_bar.n)
            
            return result
        
        # Replace the forward method temporarily
        model.model.forward = forward_with_progress
        
        try:
            # Call the original generate method
            result = original_generate(*args, **kwargs)
            # Complete the progress bar
            progress_bar.update(100 - progress_bar.n)
            progress_bar.close()
            return result
        finally:
            # Restore the original forward method
            model.model.forward = original_forward
    
    # Replace the generate method temporarily
    model.model.generate = generate_with_progress
    
    try:
        with torch.no_grad():
            print("Starting inference...")
            wav = model.inference(
                text,
                prompt_speech,
                prompt_text,
                gender,
                pitch,
                speed,
            )
            print("Inference completed successfully!")
    finally:
        # Restore the original generate method
        model.model.generate = original_generate
    
    return wav


def run_tts(
    text,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="example/results",
):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Saving audio to: {save_dir}")

    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")
    print("=" * 50)
    print(f"Processing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # Show progress bar for the overall process
    with tqdm(total=2, desc="TTS Process", position=0) as pbar:
        # Perform inference
        pbar.set_description("Generating audio")
        wav = generate(text,
                prompt_speech,
                prompt_text,
                gender,
                pitch,
                speed,)
        pbar.update(1)
        
        # Save the audio
        pbar.set_description("Saving audio")
        print(f"Saving audio to: {save_path}")
        sf.write(save_path, wav, samplerate=16000)
        pbar.update(1)
    
    print(f"Audio saved successfully at: {save_path}")
    print("=" * 50)

    logging.info(f"Audio saved at: {save_path}")

    return save_path


def build_ui(model_dir, device=0):
    
    global MODEL
    
    # Initialize model with proper device handling
    device = "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
    if MODEL is None:
        MODEL = initialize_model(model_dir, device=device)
        if device == "cuda":
            MODEL = MODEL.to(device)

    # Define callback function for voice cloning
    def voice_clone(text, prompt_text, prompt_wav_upload, prompt_wav_record):
        """
        Gradio callback to clone voice using text and optional prompt speech.
        - text: The input text to be synthesised.
        - prompt_text: Additional textual info for the prompt (optional).
        - prompt_wav_upload/prompt_wav_record: Audio files used as reference.
        """
        prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
        prompt_text_clean = None if len(prompt_text) < 2 else prompt_text

        audio_output_path = run_tts(
            text,
            prompt_text=prompt_text_clean,
            prompt_speech=prompt_speech
        )
        return audio_output_path

    # Define callback function for creating new voices
    def voice_creation(text, gender, pitch, speed):
        """
        Gradio callback to create a synthetic voice with adjustable parameters.
        - text: The input text for synthesis.
        - gender: 'male' or 'female'.
        - pitch/speed: Ranges mapped by LEVELS_MAP_UI.
        """
        pitch_val = LEVELS_MAP_UI[int(pitch)]
        speed_val = LEVELS_MAP_UI[int(speed)]
        audio_output_path = run_tts(
            text,
            gender=gender,
            pitch=pitch_val,
            speed=speed_val
        )
        return audio_output_path

    with gr.Blocks(theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="orange",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_sm,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ), css="""
    :root {
        --body-background-fill: #0f172a;
        --background-fill-primary: #1e293b;
        --background-fill-secondary: #334155;
        --color-text: #f8fafc;
        --color-accent: #3b82f6;
        --color-accent-soft: #60a5fa;
        --button-primary-background-fill: #3b82f6;
        --button-primary-background-fill-hover: #2563eb;
        --button-primary-text-color: #ffffff;
        --input-background-fill: #1e293b;
        --input-border-color: #475569;
    }
    .generate-button {
        background: linear-gradient(90deg, #3b82f6, #60a5fa) !important;
        border: none !important;
        font-weight: 500 !important;
    }
    .audio-input, .text-input, .audio-output, .voice-param {
        border-radius: 8px !important;
        border: 1px solid #334155 !important;
    }
    .tabs {
        border-bottom: 1px solid #334155 !important;
        margin-bottom: 20px !important;
    }
    .tab-selected {
        border-bottom: 2px solid #3b82f6 !important;
        font-weight: 500 !important;
    }
    .container {
        max-width: 1000px !important;
        margin: 0 auto !important;
    }
    .audio-output .waveform {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
    }
    .audio-output .waveform .waveform-fill {
        background-color: #f97316 !important;
    }
    .audio-output .audio-controls {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
    }
    body {
        background-color: #0f172a !important;
    }
    """) as demo:
        # Use HTML for centered title with stars
        gr.HTML('<div style="text-align: center; margin-bottom: 5px;"><h1 style="display: inline-flex; align-items: center; justify-content: center; margin: 0;"><span style="color: #fbbf24; margin-right: 10px;">✦</span> Spark-TTS by SparkAudio <span style="color: #fbbf24; margin-left: 10px;">✦</span></h1></div>')
        
        # Add subtitle
        gr.HTML('<div style="text-align: center; margin-bottom: 20px;"><p style="color: #94a3b8; margin: 0;">High-quality text-to-speech synthesis</p></div>')
        
        # Add model status indicator with blue left border
        model_name = "Spark-TTS-0.5B"
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        gr.HTML(f'<div style="margin: 0 auto 20px auto; width: 90%; max-width: 800px;"><div style="background-color: #1e293b; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #3b82f6; display: flex; align-items: center;"><span style="display: inline-block; width: 8px; height: 8px; background-color: #4ade80; border-radius: 50%; margin-right: 10px;"></span><span style="color: #f8fafc; font-size: 14px;">Model: {model_name} | Running on: {device_type}</span></div></div>')
        
        with gr.Tabs():
            # Voice Clone Tab  
            with gr.TabItem("Voice Clone"):
                gr.HTML(
                    '<div style="margin-top: 10px;"><h3 style="color: #60a5fa; font-weight: 500; margin-bottom: 15px;">Upload reference audio or recording</h3></div>'
                )

                with gr.Row():
                    prompt_wav_upload = gr.Audio(
                        sources="upload",
                        type="filepath",
                        label="Upload reference audio (16kHz+ recommended)",
                        elem_classes="audio-input"
                    )
                    prompt_wav_record = gr.Audio(
                        sources="microphone",
                        type="filepath",
                        label="Record your voice",
                        elem_classes="audio-input"
                    )

                gr.HTML(
                    '<div style="margin-top: 20px;"><h3 style="color: #60a5fa; font-weight: 500; margin-bottom: 15px;">Enter text to synthesize</h3></div>'
                )
                
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text to synthesize",
                        lines=3,
                        placeholder="Enter text here",
                        elem_classes="text-input"
                    )
                    prompt_text_input = gr.Textbox(
                        label="Text of reference audio (Optional)",
                        lines=3,
                        placeholder="Enter the text content of your reference audio (recommended for better results)",
                        elem_classes="text-input"
                    )

                gr.HTML(
                    '<div style="margin-top: 20px;"><h3 style="color: #60a5fa; font-weight: 500; margin-bottom: 15px;">Generated Output</h3></div>'
                )
                
                audio_output = gr.Audio(
                    label="Generated Audio",
                    autoplay=True,
                    streaming=True,
                    elem_classes="audio-output"
                )

                generate_buttom_clone = gr.Button("Generate Voice Clone", size="lg", elem_classes="generate-button")

                generate_buttom_clone.click(
                    voice_clone,
                    inputs=[
                        text_input,
                        prompt_text_input,
                        prompt_wav_upload,
                        prompt_wav_record,
                    ],
                    outputs=[audio_output],
                )

            # Voice Creation Tab
            with gr.TabItem("Voice Creation"):
                gr.HTML(
                    '<div style="margin-top: 10px;"><h3 style="color: #60a5fa; font-weight: 500; margin-bottom: 15px;">Create your own voice based on the following parameters</h3></div>'
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gender = gr.Radio(
                            choices=["male", "female"], 
                            value="male", 
                            label="Gender",
                            elem_classes="voice-param"
                        )
                        pitch = gr.Slider(
                            minimum=1, 
                            maximum=5, 
                            step=1, 
                            value=3, 
                            label="Pitch",
                            elem_classes="voice-param"
                        )
                        speed = gr.Slider(
                            minimum=1, 
                            maximum=5, 
                            step=1, 
                            value=3, 
                            label="Speed",
                            elem_classes="voice-param"
                        )
                    with gr.Column(scale=2):
                        text_input_creation = gr.Textbox(
                            label="Input Text",
                            lines=5,
                            placeholder="Enter text here",
                            value="You can generate a customized voice by adjusting parameters such as pitch and speed.",
                            elem_classes="text-input"
                        )
                        create_button = gr.Button("Create Voice", size="lg", elem_classes="generate-button")

                gr.HTML(
                    '<div style="margin-top: 20px;"><h3 style="color: #60a5fa; font-weight: 500; margin-bottom: 15px;">Generated Output</h3></div>'
                )
                
                audio_output = gr.Audio(
                    label="Generated Audio", 
                    autoplay=True, 
                    streaming=True,
                    elem_classes="audio-output"
                )
                
                create_button.click(
                    voice_creation,
                    inputs=[text_input_creation, gender, pitch, speed],
                    outputs=[audio_output],
                )

    return demo


def parse_arguments():
    """
    Parse command-line arguments such as model directory and device ID.
    """
    parser = argparse.ArgumentParser(description="Spark TTS Gradio server.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to the model directory."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (e.g., 'cpu' or 'cuda:0')."
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default=None,
        help="Server host/IP for Gradio app."
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=None,
        help="Server port for Gradio app."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Build the Gradio demo by specifying the model directory and GPU device
    demo = build_ui(
        model_dir=args.model_dir,
        device=args.device
    )

    # Launch Gradio with the specified server name and port
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port
    )