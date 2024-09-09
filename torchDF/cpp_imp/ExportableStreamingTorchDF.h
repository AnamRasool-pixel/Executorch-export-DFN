#ifndef EXPORTABLE_STREAMING_TORCH_DF_H
#define EXPORTABLE_STREAMING_TORCH_DF_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>

class ExportableStreamingTorchDF {
public:
    ExportableStreamingTorchDF(int fft_size, int hop_size, int nb_bands,
                               /* additional parameters */);
    
    void initialize();

    Eigen::MatrixXf erb_fb(const Eigen::VectorXi& widths, bool normalized = true, bool inverse = false);
    Eigen::MatrixXcf mul_complex(const Eigen::MatrixXcf& t1, const Eigen::MatrixXcf& t2);
    Eigen::MatrixXf erb(const Eigen::MatrixXcf& input_data, float erb_eps = 1e-10);
    std::pair<Eigen::MatrixXf, Eigen::MatrixXf> band_mean_norm_erb(const Eigen::MatrixXf& xs, const Eigen::MatrixXf& erb_norm_state, float alpha, float denominator = 40.0);
    std::pair<Eigen::MatrixXcf, Eigen::MatrixXf> band_unit_norm(const Eigen::MatrixXcf& xs, const Eigen::MatrixXf& band_unit_norm_state, float alpha);
    std::pair<Eigen::MatrixXcf, Eigen::MatrixXcf> frame_analysis(const Eigen::VectorXf& input_frame, const Eigen::VectorXf& analysis_mem);
    std::pair<Eigen::VectorXf, Eigen::VectorXf> frame_synthesis(const Eigen::MatrixXcf& x, const Eigen::VectorXf& synthesis_mem);
    Eigen::VectorXf forward(const Eigen::VectorXf& input_frame, const Eigen::VectorXf& states, float atten_lim_db);

private:
    int fft_size;
    int frame_size;
    int window_size;
    int window_size_h;
    int freq_size;
    float wnorm;
    int df_order;
    int lookahead;
    bool always_apply_all_stages;
    int sr;

    Eigen::VectorXf window;
    int nb_df;

    Eigen::VectorXi erb_indices;
    int nb_bands;

    Eigen::MatrixXf forward_erb_matrix;
    Eigen::MatrixXf inverse_erb_matrix;

    float alpha;

    Eigen::MatrixXcf rfft_matrix;
    Eigen::MatrixXcf irfft_matrix;

    float min_db_thresh;
    float max_db_erb_thresh;
    float max_db_df_thresh;
    float normalize_atten_lim;
    float silence_thresh;

    Eigen::VectorXf zero_gains;
    Eigen::MatrixXcf zero_coefs;

    // Additional state and parameter variables
};

#endif // EXPORTABLE_STREAMING_TORCH_DF_H
