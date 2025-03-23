import os
import re
import torch
import soundfile as sf
import logging
import argparse
import gradio as gr
import numpy as np
from datetime import datetime
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI, TASK_TOKEN_MAP
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
            # Store the global_token_ids for saving later
            global_token_ids = None
            if prompt_speech:
                # Get global_token_ids from the voice prompt
                prompt, global_token_ids = model.process_prompt(
                    text,
                    prompt_speech,
                    prompt_text
                )
            
            # Call the inference method
            wav = model.inference(
                text,
                prompt_speech,
                prompt_text,
                gender,
                pitch,
                speed,
            )
            
            # Get global_token_ids from the generated voice if we're doing controlled generation
            if gender is not None and global_token_ids is None:
                # We need to capture the generated global tokens from the model
                # In controlled generation, global tokens are generated during inference
                # Do a second inference call with a short text to extract tokens
                short_text = "This is a voice sample."
                with torch.no_grad():
                    prompt = model.process_prompt_control(gender, pitch, speed, short_text)
                    model_inputs = model.tokenizer([prompt], return_tensors="pt").to(model.device)
                    generated_ids = model.model.generate(
                        **model_inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.8,
                    )
                    # Decode the generated tokens into text
                    predicts = model.tokenizer.batch_decode(
                        [generated_ids[0][len(model_inputs.input_ids[0]):]], 
                        skip_special_tokens=True
                    )[0]
                    # Extract global token IDs
                    global_token_matches = re.findall(r"bicodec_global_(\d+)", predicts)
                    if global_token_matches:
                        global_token_ids = torch.tensor([int(token) for token in global_token_matches]).long().unsqueeze(0)
            
            print("Inference completed successfully!")
    finally:
        # Restore the original generate method
        model.model.generate = original_generate
    
    # Return both the audio and the global tokens if available
    if global_token_ids is not None:
        return wav, global_token_ids
    return wav, None


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
    voice_model_path = os.path.join(save_dir, f"{timestamp}_voice.pt")

    logging.info("Starting inference...")
    print("=" * 50)
    print(f"Processing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # Show progress bar for the overall process
    with tqdm(total=3, desc="TTS Process", position=0) as pbar:
        # Perform inference
        pbar.set_description("Generating audio")
        wav, global_tokens = generate(text,
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
        
        # Save the voice model if available
        if global_tokens is not None:
            pbar.set_description("Saving voice model")
            print(f"Saving voice model to: {voice_model_path}")
            
            # Save in compatible format - wrap the tensor in a dictionary
            # with metadata to make it compatible with test.pt
            model_dict = {
                "global_tokens": global_tokens.cpu().detach(),  # Move to CPU and detach from graph
                "created_at": timestamp,
                "model_type": "Spark-TTS-Voice",
                "version": "1.0"
            }
            
            # Add metadata for voice type if available
            if gender is not None:
                model_dict["voice_type"] = {
                    "gender": gender,
                    "pitch": pitch,
                    "speed": speed
                }
            
            torch.save(model_dict, voice_model_path)
            pbar.update(1)
        else:
            pbar.update(1)
    
    result_files = {"audio": save_path}
    if global_tokens is not None:
        result_files["voice_model"] = voice_model_path
        print(f"Voice model saved successfully at: {voice_model_path}")
    
    print(f"Audio saved successfully at: {save_path}")
    print("=" * 50)

    logging.info(f"Audio saved at: {save_path}")
    if global_tokens is not None:
        logging.info(f"Voice model saved at: {voice_model_path}")

    return result_files


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

        result_files = run_tts(
            text,
            prompt_text=prompt_text_clean,
            prompt_speech=prompt_speech
        )
        
        # Return audio file and voice model file if available
        if "voice_model" in result_files:
            return result_files["audio"], result_files["voice_model"]
        return result_files["audio"], None

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
        result_files = run_tts(
            text,
            gender=gender,
            pitch=pitch_val,
            speed=speed_val
        )
        
        # Return audio file and voice model file if available
        if "voice_model" in result_files:
            return result_files["audio"], result_files["voice_model"]
        return result_files["audio"], None

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
                
                gr.HTML(
                    '<div style="background-color: #1e293b; border-radius: 8px; padding: 12px 16px; margin-bottom: 15px; border-left: 4px solid #f97316;"><p style="color: #f8fafc; font-size: 14px; margin: 0;"><strong>Note:</strong> All generated audio files and voice models are automatically saved to the <code>example/results</code> directory. To use your saved voice model in future sessions, navigate to the "Use Saved Voice" tab and upload the .pt file.</p></div>'
                )
                
                with gr.Row():
                    audio_output = gr.Audio(
                        label="Generated Audio",
                        autoplay=True,
                        streaming=True,
                        elem_classes="audio-output"
                    )
                    voice_model_output = gr.File(
                        label="Voice Model File (PT)",
                        elem_classes="voice-model-output"
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
                    outputs=[audio_output, voice_model_output],
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
                
                gr.HTML(
                    '<div style="background-color: #1e293b; border-radius: 8px; padding: 12px 16px; margin-bottom: 15px; border-left: 4px solid #f97316;"><p style="color: #f8fafc; font-size: 14px; margin: 0;"><strong>Note:</strong> All generated audio files and voice models are automatically saved to the <code>example/results</code> directory. To use your saved voice model in future sessions, navigate to the "Use Saved Voice" tab and upload the .pt file.</p></div>'
                )
                
                with gr.Row():
                    audio_output = gr.Audio(
                        label="Generated Audio", 
                        autoplay=True, 
                        streaming=True,
                        elem_classes="audio-output"
                    )
                    voice_model_output = gr.File(
                        label="Voice Model File (PT)",
                        elem_classes="voice-model-output"
                    )

                create_button.click(
                    voice_creation,
                    inputs=[text_input_creation, gender, pitch, speed],
                    outputs=[audio_output, voice_model_output],
                )

            # Use Saved Voice Model Tab
            with gr.TabItem("Use Saved Voice"):
                gr.HTML(
                    '<div style="margin-top: 10px;"><h3 style="color: #60a5fa; font-weight: 500; margin-bottom: 15px;">Generate speech using a saved voice model</h3></div>'
                )
                
                with gr.Row():
                    saved_voice_upload = gr.File(
                        label="Upload Voice Model File (.pt)",
                        file_types=[".pt"],
                        elem_classes="voice-model-input"
                    )
                
                with gr.Row():
                    saved_voice_text = gr.Textbox(
                        label="Text to synthesize",
                        lines=5,
                        placeholder="Enter text here",
                        value="This is a test of using a saved voice model file.",
                        elem_classes="text-input"
                    )
                
                saved_voice_button = gr.Button("Generate Speech", size="lg", elem_classes="generate-button")
                
                gr.HTML(
                    '<div style="margin-top: 20px;"><h3 style="color: #60a5fa; font-weight: 500; margin-bottom: 15px;">Generated Output</h3></div>'
                )
                
                with gr.Row():
                    saved_voice_audio = gr.Audio(
                        label="Generated Audio",
                        autoplay=True,
                        streaming=True,
                        elem_classes="audio-output"
                    )
                    status_box = gr.Textbox(
                        label="Status",
                        placeholder="Status will be shown here",
                        elem_classes="status-output"
                    )
                
                def use_saved_voice(text, voice_model_path):
                    if not voice_model_path:
                        return None, "Please upload a voice model file (.pt)"
                    
                    try:
                        audio_path = synthesize_with_voice_model(text, voice_model_path)
                        return audio_path, "Voice synthesis completed successfully!"
                    except Exception as e:
                        error_msg = f"Error using saved voice model: {str(e)}"
                        print(error_msg)
                        return None, error_msg
                
                saved_voice_button.click(
                    use_saved_voice,
                    inputs=[saved_voice_text, saved_voice_upload],
                    outputs=[saved_voice_audio, status_box],
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

def load_voice_model(voice_model_path):
    """
    Load a saved voice model file (.pt) containing global tokens.
    
    Args:
        voice_model_path: Path to the voice model file
        
    Returns:
        torch.Tensor or dict: The loaded global tokens or model dictionary
    """
    if not os.path.exists(voice_model_path):
        raise FileNotFoundError(f"Voice model file not found: {voice_model_path}")
    
    print(f"Loading voice model from: {voice_model_path}")
    
    try:
        # Try to load the voice model
        loaded_data = torch.load(voice_model_path)
        
        # Check if it's a dictionary (new format) or tensor (old format)
        if isinstance(loaded_data, dict) and "global_tokens" in loaded_data:
            # New format - dictionary with metadata
            global_tokens = loaded_data["global_tokens"]
            if not isinstance(global_tokens, torch.Tensor):
                raise ValueError("Invalid voice model format: 'global_tokens' is not a tensor")
            return loaded_data
        elif isinstance(loaded_data, torch.Tensor):
            # Old format - just the tensor
            return loaded_data
        else:
            raise ValueError(f"Invalid voice model format. Expected a tensor or a dictionary with 'global_tokens'")
    except Exception as e:
        print(f"Error loading voice model: {e}")
        raise ValueError(f"Failed to load voice model: {str(e)}")

    return global_tokens


def synthesize_with_voice_model(text, voice_model_path, save_dir="example/results"):
    """
    Synthesize speech using a saved voice model.
    
    Args:
        text: Text to synthesize
        voice_model_path: Path to the voice model file
        save_dir: Directory to save the resulting audio
        
    Returns:
        str: Path to the generated audio file
    """
    global MODEL
    
    # Initialize model if not already done
    if MODEL is None:
        print("Initializing model...")
        MODEL = initialize_model(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the voice global tokens
    loaded_data = load_voice_model(voice_model_path)
    
    # Handle both new and old format
    if isinstance(loaded_data, dict) and "global_tokens" in loaded_data:
        # New format - dictionary with metadata
        global_tokens = loaded_data["global_tokens"]
        print(f"Loaded voice model: {loaded_data.get('model_type', 'Unknown')}, version: {loaded_data.get('version', 'Unknown')}")
        if "voice_type" in loaded_data:
            voice_type = loaded_data["voice_type"]
            print(f"Voice type: gender={voice_type.get('gender', 'unknown')}, pitch={voice_type.get('pitch', 'unknown')}, speed={voice_type.get('speed', 'unknown')}")
    else:
        # Old format - just the tensor
        global_tokens = loaded_data
        print("Loaded legacy format voice model")
    
    # Move to correct device
    global_tokens = global_tokens.to(MODEL.device)
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")
    
    print(f"Synthesizing speech with saved voice for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    with torch.no_grad():
        # Prepare the input tokens for the model
        global_tokens_str = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_tokens.squeeze()]
        )
        
        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens_str,
            "<|end_global_token|>",
        ]
        
        inputs = "".join(inputs)
        model_inputs = MODEL.tokenizer([inputs], return_tensors="pt").to(MODEL.device)
        
        # Generate speech using the model
        generated_ids = MODEL.model.generate(
            **model_inputs,
            max_new_tokens=3000,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
        )
        
        # Trim the output tokens to remove the input tokens
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated tokens into text
        predicts = MODEL.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract semantic token IDs from the generated text
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
            .long()
            .unsqueeze(0)
        )
        
        # Convert semantic tokens back to waveform
        wav = MODEL.audio_tokenizer.detokenize(
            global_tokens.squeeze(0),
            pred_semantic_ids.to(MODEL.device),
        )
    
    # Save the audio
    print(f"Saving audio to: {save_path}")
    sf.write(save_path, wav, samplerate=16000)
    
    print(f"Audio saved successfully at: {save_path}")
    
    return save_path

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
