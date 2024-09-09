#ifndef TORCH_DF_PIPELINE_H
#define TORCH_DF_PIPELINE_H

#include "ExportableStreamingTorchDF.h"
#include <vector>
#include <string>

class TorchDFPipeline {
public:
    TorchDFPipeline(int nb_bands = 32, int hop_size = 480, int fft_size = 960, 
                    int df_order = 5, int conv_lookahead = 2, int nb_df = 96, 
                    const std::string& model_base_dir = "DeepFilterNet3",
                    float atten_lim_db = 0.0, bool always_apply_all_stages = false, 
                    const std::string& device = "cpu");

    Eigen::VectorXf forward(const Eigen::VectorXf& input_audio, int sample_rate);

private:
    int hop_size;
    int fft_size;
    int sample_rate;
    ExportableStreamingTorchDF torch_streaming_model;
    Eigen::VectorXf states;
    float atten_lim_db;

    void init_df(const std::string& model_base_dir);
};

#endif // TORCH_DF_PIPELINE_H
