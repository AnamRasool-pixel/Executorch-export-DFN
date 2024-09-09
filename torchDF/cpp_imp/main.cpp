#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include "AudioFile.h"

const int sample_rate = 48000;
const int hop_size = 480;
const int fft_size = 960;

const std::string model_path = "/mnt/d/puretorch/DeepFilterNet/torchDF/final_ETDF.pt";
const std::string input_audio_path = "/mnt/d/puretorch/DeepFilterNet/torchDF/examples/testAudio.wav";
const std::string output_audio_path = "/mnt/d/puretorch/DeepFilterNet/torchDF/examples/Audio_enhanced.wav";

void printTensorInfo(const torch::Tensor& tensor, const std::string& name) {
    std::cout << name << " - Type: " << tensor.dtype() << ", Shape: " << tensor.sizes() << std::endl;
}

int main() {
    try {
        // Load the model
        torch::jit::script::Module torch_streaming_model = torch::jit::load(model_path);
        torch_streaming_model.to(torch::kCPU);
        std::cout << "Model loaded successfully." << std::endl;

        // Print model attributes
        std::cout << "Model attributes:" << std::endl;
        for (const auto& attr : torch_streaming_model.named_attributes()) {
            std::cout << attr.name << " - Type: " << attr.value.type()->str() << std::endl;
        }

        // Initialize states and attenuation limit
        auto states_full_len_tensor = torch_streaming_model.attr("states_full_len").toTensor();
        printTensorInfo(states_full_len_tensor, "states_full_len_tensor");
        int64_t states_full_len = states_full_len_tensor.item<int64_t>();
        std::cout << "states_full_len: " << states_full_len << std::endl;

        auto states = torch::zeros({states_full_len}, torch::kFloat32);
        auto atten_lim_db = torch::tensor(0.0f, torch::kFloat32);

        printTensorInfo(states, "states");
        printTensorInfo(atten_lim_db, "atten_lim_db");

        // Load and preprocess input audio
        AudioFile<float> audioFile;
        if (!audioFile.load(input_audio_path)) {
            std::cerr << "Failed to load audio file: " << input_audio_path << std::endl;
            return -1;
        }

        if (audioFile.getSampleRate() != sample_rate) {
            std::cerr << "Expected sample rate " << sample_rate << ", but got " << audioFile.getSampleRate() << std::endl;
            return -1;
        }

        // Convert audio samples to a tensor and process it
        auto samples = audioFile.samples[0];  // Assuming mono audio for simplicity
        auto noisy_audio = torch::tensor(samples, torch::kFloat32).unsqueeze(0);
        printTensorInfo(noisy_audio, "noisy_audio");

        // Preprocess audio
        auto input_audio = noisy_audio.squeeze(0);
        auto orig_len = input_audio.size(0);

        auto hop_size_divisible_padding_size = (hop_size - orig_len % hop_size) % hop_size;
        orig_len += hop_size_divisible_padding_size;
        input_audio = torch::nn::functional::pad(input_audio, torch::nn::functional::PadFuncOptions({0, fft_size + hop_size_divisible_padding_size}));
        printTensorInfo(input_audio, "input_audio");

        auto chunked_audio = input_audio.split(hop_size);

        torch_streaming_model.eval();

        // Enhance audio
        std::vector<torch::Tensor> output_frames;
        for (size_t i = 0; i < chunked_audio.size(); ++i) {
            try {
                const auto& input_frame = chunked_audio[i];
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_frame);
                inputs.push_back(states);
                inputs.push_back(atten_lim_db);

                std::cout << "Processing frame " << i << ":" << std::endl;
                printTensorInfo(input_frame, "input_frame");
                printTensorInfo(states, "states");
                printTensorInfo(atten_lim_db, "atten_lim_db");

                auto output = torch_streaming_model.forward(inputs).toTuple();

                std::cout << "Output types:" << std::endl;
                for (size_t j = 0; j < output->elements().size(); ++j) {
                    std::cout << "Output " << j << " - Type: " << output->elements()[j].type()->str() << std::endl;
                }

                auto enhanced_audio_frame = output->elements()[0].toTensor();
                states = output->elements()[1].toTensor();
                printTensorInfo(enhanced_audio_frame, "enhanced_audio_frame");
                printTensorInfo(states, "updated_states");

                output_frames.push_back(enhanced_audio_frame);

            } catch (const c10::Error& e) {
                std::cerr << "Error processing frame: " << e.what() << std::endl;
                break;
            }
        }

        // ... (rest of the code remains the same)

    } catch (const c10::Error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}