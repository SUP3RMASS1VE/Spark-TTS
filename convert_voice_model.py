#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert Spark-TTS Voice Model Files to the new format.

This script converts old-format voice model files (raw tensors)
to the new format (dictionary with metadata).

Usage:
    python convert_voice_model.py input_file.pt output_file.pt
"""

import os
import sys
import torch
from datetime import datetime

def convert_voice_model(input_path, output_path, gender=None, pitch=None, speed=None):
    """
    Convert a voice model file from old format to new format.
    
    Args:
        input_path: Path to the input voice model file
        output_path: Path to save the converted voice model file
        gender: Optional gender (male/female)
        pitch: Optional pitch level
        speed: Optional speed level
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    print(f"Loading voice model from: {input_path}")
    
    try:
        # Load the input model
        loaded_data = torch.load(input_path)
        
        # Check if it's already in the new format
        if isinstance(loaded_data, dict) and "global_tokens" in loaded_data:
            print("File is already in the new format. No conversion needed.")
            if output_path != input_path:
                print(f"Copying to: {output_path}")
                torch.save(loaded_data, output_path)
            return True
        
        # If it's a tensor, convert to new format
        if isinstance(loaded_data, torch.Tensor):
            global_tokens = loaded_data
            
            # Create new model dictionary
            model_dict = {
                "global_tokens": global_tokens.cpu().detach() if hasattr(global_tokens, "detach") else global_tokens,
                "created_at": datetime.now().strftime("%Y%m%d%H%M%S"),
                "model_type": "Spark-TTS-Voice",
                "version": "1.0",
                "converted_from": os.path.basename(input_path)
            }
            
            # Add voice type if provided
            if gender is not None:
                model_dict["voice_type"] = {
                    "gender": gender,
                    "pitch": pitch,
                    "speed": speed
                }
            
            # Save the converted model
            print(f"Saving converted model to: {output_path}")
            torch.save(model_dict, output_path)
            print("Conversion successful!")
            return True
        else:
            print(f"Error: Unsupported format. Expected a tensor or a dictionary.")
            return False
            
    except Exception as e:
        print(f"Error converting voice model: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_voice_model.py input_file.pt output_file.pt [gender] [pitch] [speed]")
        print("\nExample:")
        print("  python convert_voice_model.py jad.pt jad_converted.pt male moderate moderate")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Optional parameters
    gender = sys.argv[3] if len(sys.argv) > 3 else None
    pitch = sys.argv[4] if len(sys.argv) > 4 else None
    speed = sys.argv[5] if len(sys.argv) > 5 else None
    
    convert_voice_model(input_path, output_path, gender, pitch, speed)

if __name__ == "__main__":
    main() 