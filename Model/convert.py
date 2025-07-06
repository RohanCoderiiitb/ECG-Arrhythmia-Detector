# Save this code as, for example, convert_to_c_array.py

import os

def tflite_to_c_array(tflite_file_path, output_header_path):
    """
    Converts a TensorFlow Lite model file into a C header file containing a byte array.

    Args:
        tflite_file_path (str): Path to the input .tflite model file.
        output_header_path (str): Path where the output .h header file will be saved.
    """
    try:
        with open(tflite_file_path, 'rb') as f_tflite:
            tflite_model_bytes = f_tflite.read()

        # Determine the variable name from the tflite file name
        base_name = os.path.basename(tflite_file_path)
        var_name = os.path.splitext(base_name)[0].replace('.', '_').replace('-', '_') + '_tflite'

        with open(output_header_path, 'w') as f_header:
            f_header.write(f'#ifndef {var_name.upper()}_H\n')
            f_header.write(f'#define {var_name.upper()}_H\n\n')
            f_header.write(f'const unsigned char {var_name}[] = {{\n')

            for i, byte in enumerate(tflite_model_bytes):
                f_header.write(f'0x{byte:02x}')
                if i < len(tflite_model_bytes) - 1:
                    f_header.write(', ')
                if (i + 1) % 12 == 0:  # 12 bytes per line for readability
                    f_header.write('\n  ')
            f_header.write('\n};\n')
            f_header.write(f'const unsigned int {var_name}_len = {len(tflite_model_bytes)};\n\n')
            f_header.write(f'#endif // {var_name.upper()}_H\n')

        print(f"Successfully converted '{tflite_file_path}' to '{output_header_path}'")
        print(f"The C array variable name is: {var_name}")
        print(f"The length variable name is: {var_name}_len")

    except FileNotFoundError:
        print(f"Error: File not found at {tflite_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- How to use this script ---
if __name__ == "__main__":
    # Make sure 'my_ecg_model.tflite' is in the same directory as this script,
    # or provide the full path to your .tflite file.
    tflite_input_file = 'ecg-arrhythmia-model.tflite'
    c_header_output_file = 'model_data.h'

    tflite_to_c_array(tflite_input_file, c_header_output_file)