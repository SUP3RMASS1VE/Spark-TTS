# Spark-TTS: Text-to-Speech Synthesis  
![Screenshot 2025-03-23 135635](https://github.com/user-attachments/assets/f996a3af-283c-4a53-9b19-7ab45256c944)
![Screenshot 2025-03-23 135719](https://github.com/user-attachments/assets/3cadfb81-1c37-4fbd-900b-a3840a0857e6)


Spark-TTS is a high-quality text-to-speech synthesis system that enables voice cloning and speech generation using deep learning models. The system allows users to generate speech with customizable parameters such as pitch, speed, and gender. The results may vary, and the generated speech is typically around **30 seconds long**.  

## Features  
- **Voice Cloning**: Generate speech using a reference audio sample.  
- **Custom Voice Creation**: Adjust pitch, speed, and gender for unique voices.  
- **Create & Save Voice Models**: Generate a voice model once and reuse it without needing an audio sample every time.  
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

