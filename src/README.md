# 🎯 CTC Forced Aligner

We are open-sourcing the CTC forced aligner used in [Deskpai](https://www.deskpai.com).

With focuses on production-ready model inference, it supports 18 different alignment models, including multilingual models(German, English, Spanish, French and Italian etc), and provides SRT and WebVTT alignment and generation out of box. It supports both ONNXRuntime and PyTorch for model serving.

## 🚀 Installation  

- CPU inference via ONNXRuntime

```bash
pip install ctc_forced_aligner
```

- GPU inference via ONNXRuntime

```bash
pip install ctc_forced_aligner[gpu]
```

- CPU/GPU inference via PyTorch

```bash
pip install ctc_forced_aligner[torch]
```

- Install all dependencies

```bash
pip install ctc_forced_aligner[all]
```

## 📝 Sample Inference Code  

- CPU/GPU inference via ONNXRuntime

```python
from ctc_forced_aligner import AlignmentSingleton
alignment_service = AlignmentSingleton()
input_audio_path = "audio.mp3"
input_text_path = "input.txt"
output_srt_path = "output.srt"
ret = alignment_service.generate_srt(input_audio_path,
                                     input_text_path,
                                     output_srt_path)
if ret:
    print(f"Aligned SRT is generated at {output_srt_path}")
output_vtt_path = "output.vtt"
ret = alignment_service.generate_webvtt(input_audio_path,
                                        input_text_path,
                                        output_vtt_path)
if ret:
    print(f"aligned VTT is generated to {output_vtt_path}")
```

- CPU/GPU inference via PyTorch

```python
from ctc_forced_aligner import AlignmentTorch
at = AlignmentTorch()
ret = at.generate_srt(input_audio_path, input_text_path, output_srt_path)
if ret:
    print(f"aligned srt is generated to {output_srt_path}")
ret = at.generate_webvtt(input_audio_path, input_text_path, output_vtt_path)
if ret:
    print(f"aligned VTT is generated to {output_vtt_path}")
```

- Inference with multiple models

```python
from ctc_forced_aligner import AlignmentTorch
at = AlignmentTorch()
ret = at.generate_srt(input_audio_path, input_text_path, output_srt_path, model_type='WAV2VEC2_ASR_BASE_960H')
if ret:
    print(f"aligned srt is generated to {output_srt_path}")
ret = at.generate_webvtt(input_audio_path, input_text_path, output_vtt_path, model_type='WAV2VEC2_ASR_BASE_960H')
if ret:
    print(f"aligned VTT is generated to {output_vtt_path}")
```

## Models Supported

### ✅ Wav2Vec2 Models

These are fine-tuned models with a **CTC-based ASR head**:
- `WAV2VEC2_ASR_BASE_960H`
- `WAV2VEC2_ASR_BASE_100H`
- `WAV2VEC2_ASR_BASE_10M`
- `WAV2VEC2_ASR_LARGE_10M`
- `WAV2VEC2_ASR_LARGE_100H`
- `WAV2VEC2_ASR_LARGE_960H`
- `WAV2VEC2_ASR_LARGE_LV60K_10M`
- `WAV2VEC2_ASR_LARGE_LV60K_100H`
- `WAV2VEC2_ASR_LARGE_LV60K_960H`

### ✅ VoxPopuli ASR Models (Multilingual)

These models are fine-tuned for **specific languages**:
- `VOXPOPULI_ASR_BASE_10K_DE` (German ASR)
- `VOXPOPULI_ASR_BASE_10K_EN` (English ASR)
- `VOXPOPULI_ASR_BASE_10K_ES` (Spanish ASR)
- `VOXPOPULI_ASR_BASE_10K_FR` (French ASR)
- `VOXPOPULI_ASR_BASE_10K_IT` (Italian ASR)

- Fine-tuned on **VoxPopuli** speech corpus.

### ✅ HuBERT ASR Models
- `HUBERT_ASR_LARGE`
- `HUBERT_ASR_XLARGE`

✅ Characteristics:
- Based on **HuBERT** (Hidden-Unit BERT), which learns speech representations differently from Wav2Vec2.
- Has an ASR head and works similarly to Wav2Vec2 ASR models.
- `XLARGE` is a bigger model with more parameters for better accuracy.

## 💡 Which One and How to Use?

**For PyTorch serving**, use `AlignmentTorch` or `AlignmentTorchSingleton`. 

- **For English ASR** → `WAV2VEC2_ASR_LARGE_960H` or `HUBERT_ASR_LARGE`
- **For multilingual ASR** → `VOXPOPULI_ASR_BASE_10K_*`
- **For low-resource ASR** → `WAV2VEC2_ASR_BASE_10M` (smallest model)
- **For best accuracy** → `WAV2VEC2_ASR_LARGE_LV60K_960H` or `HUBERT_ASR_XLARGE`

**For ONNXRuntime serving** with minimum dependencies, use `Alignment` or `AlignmentSingleton`.

Please contact [us](mailto:dev@deskpai.com) if you want to integrate your model into this package.

## 📄 License

### Code

- This project includes code from [pytorch/audio](https://github.com/pytorch/audio), licensed under the `BSD-2-Clause` license.
- This project includes code from [MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner), licensed under the `BSD` license.`This project is licensed under the BSD License, note that the default model has CC-BY-NC 4.0 License, so make sure to use a different model for commercial usage.`
- Modifications and additional code are contributed by [Deskpai.com](https://www.deskpai.com) and licensed under the [DOSL-1.0 license](https://github.com/deskpai/deskpai/blob/main/LICENSE).

### Model

- The following models are developed by Meta AI (formerly Facebook AI) under `MIT License` and redistributed with the same license.
  - `WAV2VEC2_ASR_BASE_960H`
  - `WAV2VEC2_ASR_BASE_100H`
  - `WAV2VEC2_ASR_BASE_10M`
  - `WAV2VEC2_ASR_LARGE_10M`
  - `WAV2VEC2_ASR_LARGE_100H`
  - `WAV2VEC2_ASR_LARGE_960H`
  - `WAV2VEC2_ASR_LARGE_LV60K_10M`
  - `WAV2VEC2_ASR_LARGE_LV60K_100H`
  - `WAV2VEC2_ASR_LARGE_LV60K_960H`
- VoxPopuli and HuBERT models are also developed by Meta AI and are generally released under the MIT License. The specific licensing for these models can be found in their respective repositories or documentation. Please check it on your own.
  - `VOXPOPULI_ASR_BASE_10K_DE`
  - `VOXPOPULI_ASR_BASE_10K_EN`
  - `VOXPOPULI_ASR_BASE_10K_ES`
  - `VOXPOPULI_ASR_BASE_10K_FR`
  - `VOXPOPULI_ASR_BASE_10K_IT`
  - `HUBERT_ASR_LARGE`
  - `HUBERT_ASR_XLARGE`
- The model `MMS_FA` is published by the authors of Scaling Speech Technology to 1,000+ Languages Pratap et al., 2023 under `CC-BY-NC 4.0 License`.
- The onnx model weights are created by [Deskpai.com](https://www.deskpai.com) based on the model of [MahmoudAshraf/mms-300m-1130-forced-aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) and under `CC-BY-NC 4.0 License`.

📝 Note: It's essential to verify the licensing terms from the official repositories or documentation before using these models. 

## 🙏 Reference

- [LESS PEAKY AND MORE ACCURATE CTC FORCED ALIGNMENT BY LABEL PRIORS](https://arxiv.org/pdf/2406.02560)
- [Montreal Forced Aligner User Guide](https://montreal-forced-aligner.readthedocs.io/en/stable/user_guide/index.html)
- [Forced Alignment with Wav2Vec2](https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html)
- [NeuFA: Neural Network Based End-to-End Forced Aligner](https://arxiv.org/abs/2203.16838)
- [Tradition or Innovation: A Comparison of Modern ASR Methods for Forced
Alignment](https://arxiv.org/pdf/2406.19363v1)
