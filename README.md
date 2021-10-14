# DeepFilterNet
A Low Complexity Speech Enhancement Framework for Full-Band Audio (48kHz) based on Deep Filtering.
Audio samples from the voice bank/DEMAND test set can found at https://rikorose.github.io/DeepFilterNet-Samples/

* `libDF` contains Rust code used for data loading and augmentation.
* `DeepFilterNet` contains Python code including a libDF wrapper for data loading, DeepFilterNet training, testing and visualization.
* `models` contains DeepFilterNet model weights and config.

## Usage

System requirements are `cargo` and `pip` (Rust and Python package managers).
Usage of a `conda` or `virtualenv` recommended.
This framework is currently only tested under Linux.

Installation of python dependencies and libDF:
```bash
cd path/to/DeepFilterNet/  # cd into repository
# Recommended: Install or activate a python env
pip install maturin  # Used to compile libDF and load
# A) Build python wheel manually and install using pip
maturin build --release -m DeepFilterNet/Cargo.toml
# Install python wheel. Make sure to specify the correct DeepFilterNet and python version
pip install target/wheels/DeepFilterNet-0.1.0-cp39-cp39-linux_x86_64.whl
# B) Directly install into env via maturin develop: maturin develop --release -m DeepFilterNet/Cargo.toml
# Optional: Install cuda version of pytorch from pytorch.org
pip install -r requirements.txt  # Install remaining dependencies
```

To enhance noisy audio files using DeepFilterNet run
```bash
# usage: enhance.py [-h] [--output-dir OUTPUT_DIR] model_base_dir noisy_audio_files [noisy_audio_files ...]
python DeepFilterNet/df/enhance.py models/DeepFilterNet/ path/to/noisy_audio.wav
```

## Citation

This code accompanies the paper 'DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering'.

```bibtex
@misc{schröter2021deepfilternet,
      title={DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering}, 
      author={Hendrik Schröter and Alberto N. Escalante-B. and Tobias Rosenkranz and Andreas Maier},
      year={2021},
      eprint={2110.05588},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

## License

DeepFilterNet is free and open source! All code in this repository is dual-licensed under either:

* MIT License ([LICENSE-MIT](docs/LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](docs/LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))

at your option. This means you can select the license you prefer!

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
