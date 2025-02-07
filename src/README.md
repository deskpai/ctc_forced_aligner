# 🎯 CTC Forced Aligner



We are open-sourcing the CTC forced aligner used in [Deskpai](https://www.deskpai.com), which supports both `GPU` and `CPU` via `onnxruntime` without relying on heavy libraries like `torch` or `transformers`.  

## 🚀 Installation  

- Install from pypi

```bash
pip install ctc_forced_aligner
```

- Install from ssh+https

```
pip install git+https://github.com/deskpai/ctc_forced_aligner.git
```

- Install from local (dev)

```
git checkout https://github.com/deskpai/ctc_forced_aligner.git
cd ctc_forced_aligner/src
pip install -e .
```


## 📝 Sample Code  

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
```


## 🙏 Special Thanks  

- The model weights are adapted from [MahmoudAshraf/mms-300m-1130-forced-aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner), trained by [Mahmoud Ashraf](https://huggingface.co/MahmoudAshraf). We removed heavy dependencies (`uroman`, `torch` etc) and added new features (SRT output, etc.) to make it production-ready.  

- Ruizhe Huang's paper, [LESS PEAKY AND MORE ACCURATE CTC FORCED ALIGNMENT BY LABEL PRIORS](https://arxiv.org/pdf/2406.02560), provided valuable insights into the model.  
