#include <EloquentTinyML.h>
// Include your converted model file
#include "model.h"

// --------------------------------------------------------------------------
// CONFIGURATION
// --------------------------------------------------------------------------
#define N_INPUTS    11  // The 11 temporal features
#define N_OUTPUTS   5   // Standard MIT-BIH classes: N, S, V, F, Q
#define ARENA_SIZE  60 * 1024 // 60KB reserved for TFLite operations

// Define the class labels matching your training mapping
const char* CLASSES[] = {
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular (V)",
    "Fusion (F)",
    "Unknown (Q)"
};

// Initialize the TinyML Interpreter
Eloquent::TinyML::TfLite<N_INPUTS, N_OUTPUTS, ARENA_SIZE> ml;

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("___ECG Arrhythmia Detector (ESP32)___");

    // Initialize the model
    if (!ml.begin(ecg_arrhythmia_model_tflite)) {
        Serial.println("Cannot init model");
        Serial.println(ml.getErrorMessage());
        while (true) delay(1000);
    }
    
    Serial.println("Model loaded successfully.");
    Serial.println("Send 11 CSV float features to classify.");
}

void loop() {
    // 1. Check if data is available from Python script
    if (Serial.available()) {
        float input_features[N_INPUTS];
        
        // 2. Parse 11 floats from Serial
        // format expected: "0.12, 0.45, 0.99, ... \n"
        for (int i = 0; i < N_INPUTS; i++) {
            input_features[i] = Serial.parseFloat(); 
        }

        // Clear the buffer (consume newline)
        while(Serial.available() > 0 && Serial.peek() < '0') {
            Serial.read();
        }

        // 3. Run Inference
        float prediction[N_OUTPUTS] = {0};
        ml.predict(input_features, prediction);

        // 4. Find the class with highest probability (argmax)
        int max_index = 0;
        float max_prob = prediction[0];
        
        for (int i = 1; i < N_OUTPUTS; i++) {
            if (prediction[i] > max_prob) {
                max_prob = prediction[i];
                max_index = i;
            }
        }

        // 5. Output result back to Serial
        Serial.print("Class: ");
        Serial.print(CLASSES[max_index]);
        Serial.print(" | Confidence: ");
        Serial.println(max_prob);
    }
}
