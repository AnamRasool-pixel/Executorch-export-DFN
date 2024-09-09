#include "TorchDFPipeline.h"
#include <iostream>

TorchDFPipeline::TorchDFPipeline(int nb_bands, int hop_size, int fft_size, int df_order, 
                                 int conv_lookahead, int nb_df, const std::string& model_base_dir, 
                                 float atten_lim_db, bool always_apply_all_stages, const std::string& device)
    : hop_size(hop_size), fft_size(fft_size), atten_lim_db(atten_lim_db), sample_rate(48000) {
    
    init_df(model_base_dir);
    torch_streaming_model = ExportableStreamingTorchDF(fft_size, hop_size, nb_bands, /* add other parameters */);
    torch_streaming_model.initialize();

    states = Eigen::VectorXf::Zero(torch_streaming_model.states_full_len);
}

void TorchDFPipeline::init_df(const std::string& model_base_dir) {
    // Load the model weights from checkpoint and initialize the model components
}

Eigen::VectorXf TorchDFPipeline::forward(const Eigen::VectorXf& input_audio, int sample_rate) {
    if (input_audio.size() == 0 || sample_rate != this->sample_rate) {
        std::cerr << "Invalid input audio or sample rate!" << std::endl;
        return Eigen::VectorXf();
    }

    int orig_len = input_audio.size();
    int num_chunks = (orig_len + hop_size - 1) / hop_size;
    std::vector<Eigen::VectorXf> output_frames;

    for (int i = 0; i < num_chunks; ++i) {
        int start_idx = i * hop_size;
        int end_idx = std::min((i + 1) * hop_size, orig_len);
        Eigen::VectorXf input_frame = input_audio.segment(start_idx, end_idx - start_idx);

        Eigen::VectorXf enhanced_audio_frame;
        Eigen::VectorXf new_states;
        Eigen::VectorXf lsnr;

        std::tie(enhanced_audio_frame, new_states, lsnr) = torch_streaming_model.forward(input_frame, states, atten_lim_db);

        output_frames.push_back(enhanced_audio_frame);
        states = new_states;
    }

    Eigen::VectorXf enhanced_audio(orig_len);
    for (size_t i = 0; i < output_frames.size(); ++i) {
        enhanced_audio.segment(i * hop_size, output_frames[i].size()) = output_frames[i];
    }

    return enhanced_audio;
}
