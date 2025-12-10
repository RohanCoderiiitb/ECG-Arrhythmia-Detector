#include <EloquentTinyML.h>
#include "model.h"

#define N_INPUTS    11  
#define N_OUTPUTS   5   
#define ARENA_SIZE  60 * 1024 

const char* CLASSES[] = {
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular (V)",
    "Fusion (F)",
    "Unknown (Q)"
};

Eloquent::TinyML::TfLite<N_INPUTS, N_OUTPUTS, ARENA_SIZE> ml;

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("___ECG Arrhythmia Detector (ESP32)___");

    if (!ml.begin(ecg_arrhythmia_model_tflite)) {
        Serial.println("Cannot init model");
        Serial.println(ml.getErrorMessage());
        while (true) delay(1000);
    }
    
    Serial.println("Model loaded successfully.");
    Serial.println("Send 11 CSV float features to classify.");
}

void loop() {
    if (Serial.available()) {
        float input_features[N_INPUTS];
        
        for (int i = 0; i < N_INPUTS; i++) {
            input_features[i] = Serial.parseFloat(); 
        }

        while(Serial.available() > 0 && Serial.peek() < '0') {
            Serial.read();
        }

        float prediction[N_OUTPUTS] = {0};
        ml.predict(input_features, prediction);

        int max_index = 0;
        float max_prob = prediction[0];
        
        for (int i = 1; i < N_OUTPUTS; i++) {
            if (prediction[i] > max_prob) {
                max_prob = prediction[i];
                max_index = i;
            }
        }
        Serial.print("Class: ");
        Serial.print(CLASSES[max_index]);
        Serial.print(" | Confidence: ");
        Serial.println(max_prob);
    }
}
