#include "ExportableStreamingTorchDF.h"
#include <unsupported/Eigen/FFT>
#include <iostream>

ExportableStreamingTorchDF::ExportableStreamingTorchDF(int fft_size, int hop_size, int nb_bands, 
                                                       /* additional parameters */)
    : fft_size(fft_size), frame_size(hop_size), window_size(fft_size), window_size_h(fft_size / 2),
      freq_size(fft_size / 2 + 1), wnorm(1.0 / (window_size * window_size / (2 * frame_size))),
      df_order(5), lookahead(2), always_apply_all_stages(false), sr(48000), nb_bands(nb_bands),
      alpha(0.99), min_db_thresh(-10.0), max_db_erb_thresh(30.0), max_db_df_thresh(20.0),
      normalize_atten_lim(20.0), silence_thresh(1e-7), nb_df(96) {
    
    window = Eigen::VectorXf(fft_size);
    for (int i = 0; i < fft_size; ++i) {
        float val = std::sin(0.5 * M_PI * (i + 0.5) / window_size_h);
        window[i] = std::sin(0.5 * M_PI * val * val);
    }

    erb_indices = Eigen::VectorXi(32);
    int erb_idx_vals[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 
                          10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67};
    for (int i = 0; i < 32; ++i) {
        erb_indices[i] = erb_idx_vals[i];
    }

    forward_erb_matrix = erb_fb(erb_indices, true, false);
    inverse_erb_matrix = erb_fb(erb_indices, true, true);

    // Initialize FFT matrices, zero gains, and zero coefficients
    rfft_matrix = Eigen::MatrixXcf::Zero(window_size, freq_size);
    irfft_matrix = rfft_matrix.transpose().conjugate();

    zero_gains = Eigen::VectorXf::Zero(nb_bands);
    zero_coefs = Eigen::MatrixXcf::Zero(df_order, nb_df);
}

Eigen::MatrixXf ExportableStreamingTorchDF::erb_fb(const Eigen::VectorXi& widths, bool normalized, bool inverse) {
    int n_freqs = widths.sum();
    Eigen::VectorXf all_freqs = Eigen::VectorXf::LinSpaced(n_freqs, 0, sr / 2);

    Eigen::VectorXi b_pts(widths.size());
    b_pts[0] = 0;
    for (int i = 1; i < widths.size(); ++i) {
        b_pts[i] = b_pts[i-1] + widths[i-1];
    }

    Eigen::MatrixXf fb = Eigen::MatrixXf::Zero(n_freqs, widths.size());
    for (int i = 0; i < widths.size(); ++i) {
        fb.block(b_pts[i], i, widths[i], 1).setOnes();
    }

    if (inverse) {
        fb.transposeInPlace();
        if (!normalized) {
            fb.array().colwise() /= fb.rowwise().sum().array();
        }
    } else {
        if (normalized) {
            fb.array().rowwise() /= fb.colwise().sum().array();
        }
    }

    return fb;
}

Eigen::MatrixXcf ExportableStreamingTorchDF::mul_complex(const Eigen::MatrixXcf& t1, const Eigen::MatrixXcf& t2) {
    Eigen::MatrixXcf result(t1.rows(), t1.cols());
    for (int i = 0; i < t1.rows(); ++i) {
        result(i, 0) = t1(i, 0).real() * t2(i, 0).real() - t1(i, 0).imag() * t2(i, 0).imag();
        result(i, 1) = t1(i, 0).real() * t2(i, 0).imag() + t1(i, 0).imag() * t2(i, 0).real();
    }
    return result;
}

Eigen::MatrixXf ExportableStreamingTorchDF::erb(const Eigen::MatrixXcf& input_data, float erb_eps) {
    Eigen::MatrixXf magnitude_squared = input_data.array().square().matrix().rowwise().sum();
    Eigen::MatrixXf erb_features = magnitude_squared * forward_erb_matrix;
    Eigen::MatrixXf erb_features_db = 10.0 * (erb_features.array() + erb_eps).log10();
    return erb_features_db;
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> ExportableStreamingTorchDF::band_mean_norm_erb(const Eigen::MatrixXf& xs, const Eigen::MatrixXf& erb_norm_state, float alpha, float denominator) {
    Eigen::MatrixXf new_erb_norm_state = xs * alpha + erb_norm_state * (1.0 - alpha);
    Eigen::MatrixXf output = (xs - new_erb_norm_state) / denominator;
    return std::make_pair(output, new_erb_norm_state);
}

std::pair<Eigen::MatrixXcf, Eigen::MatrixXf> ExportableStreamingTorchDF::band_unit_norm(const Eigen::MatrixXcf& xs, const Eigen::MatrixXf& band_unit_norm_state, float alpha) {
    Eigen::MatrixXf xs_abs = xs.array().abs().matrix().rowwise().norm();
    Eigen::MatrixXf new_band_unit_norm_state = xs_abs * alpha + band_unit_norm_state * (1.0 - alpha);
    Eigen::MatrixXcf output = xs.array().colwise() / new_band_unit_norm_state.array().sqrt();
    return std::make_pair(output, new_band_unit_norm_state);
}

std::pair<Eigen::MatrixXcf, Eigen::MatrixXcf> ExportableStreamingTorchDF::frame_analysis(const Eigen::VectorXf& input_frame, const Eigen::VectorXf& analysis_mem) {
    Eigen::VectorXf buf = analysis_mem;
    buf.segment(analysis_mem.size(), input_frame.size()) = input_frame;
    buf = buf.array() * window.array();
    
    Eigen::FFT<float> fft;
    Eigen::MatrixXcf rfft_buf(freq_size, 2);
    fft.fwd(rfft_buf, buf);
    
    return std::make_pair(rfft_buf * wnorm, input_frame);
}

std::pair<Eigen::VectorXf, Eigen::VectorXf> ExportableStreamingTorchDF::frame_synthesis(const Eigen::MatrixXcf& x, const Eigen::VectorXf& synthesis_mem) {
    Eigen::FFT<float> fft;
    Eigen::VectorXf x_real(window_size);
    fft.inv(x_real, x);
    
    x_real = x_real.array() * window.array() * fft_size;
    Eigen::VectorXf x_first = x_real.head(frame_size);
    Eigen::VectorXf x_second = x_real.tail(window_size - frame_size);

    Eigen::VectorXf output = x_first + synthesis_mem;
    return std::make_pair(output, x_second);
}

Eigen::VectorXf ExportableStreamingTorchDF::forward(const Eigen::VectorXf& input_frame, const Eigen::VectorXf& states, float atten_lim_db) {
    // Unpack states, perform forward computation, and pack states
    // This function would need to be implemented similarly to the provided PyTorch code
}

#endif // EXPORTABLE_STREAMING_TORCH_DF_H
