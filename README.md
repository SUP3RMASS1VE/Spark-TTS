# Spark-TTS: Text-to-Speech Synthesis  
![Screenshot 2025-03-12 191533](https://github.com/user-attachments/assets/1076221d-73bc-47d4-9a74-1fe7d7d04397)  

Spark-TTS is a high-quality text-to-speech synthesis system that enables voice cloning and speech generation using deep learning models. The system allows users to generate speech with customizable parameters such as pitch, speed, and gender. The results may vary, and the generated speech is typically around **30 seconds long**.  

## Features  
- **Voice Cloning**: Generate speech using a reference audio sample.  
- **Custom Voice Creation**: Adjust pitch, speed, and gender for unique voices.  
- **Gradio UI**: User-friendly web interface for generating and saving audio.  
- **GPU Support**: Utilizes CUDA when available for faster processing.  

## Installation  
Install via [Pinokio](https://pinokio.computer).  

## Expected Output  
- Generated speech audio in `.wav` format.  
- **Note**: The output duration varies, but is generally around **30 seconds**.  

## Model Download  
The required model is automatically downloaded from Hugging Face if not present locally.  

## Acknowledgments  
- Built with PyTorch, Hugging Face, and Gradio.  
- Model based on SparkAudio's Spark-TTS-0.5B.  

## License  
MIT License. See `LICENSE` for more details.  

---
