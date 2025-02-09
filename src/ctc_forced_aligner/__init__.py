import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple

import librosa
import numpy
import onnxruntime
import requests

from .norm_config import norm_config

SAMPLING_FREQ = 16000  # Define the expected sampling frequency
VOCAB_DICT = {'<blank>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, 'a': 4, 'i': 5, 'e': 6, 'n': 7, 'o': 8, 'u': 9, 't': 10,
              's': 11, 'r': 12, 'm': 13, 'k': 14, 'l': 15, 'd': 16, 'g': 17, 'h': 18, 'y': 19, 'b': 20, 'p': 21,
              'w': 22, 'c': 23, 'v': 24, 'j': 25, 'z': 26, 'f': 27, "'": 28, 'q': 29, 'x': 30}
MODEL_URL = 'https://huggingface.co/deskpai/ctc_forced_aligner/resolve/main/04ac86b67129634da93aea76e0147ef3.onnx'

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
logger.debug(__file__)

__all__ = ['Alignment', 'AlignmentSingleton']


class Tokenizer:
    """Standalone Tokenizer for Wav2Vec2 CTC Forced Alignment"""

    def __init__(self):
        self.vocab_dict = VOCAB_DICT
        self.id_to_token = {v: k for k, v in self.vocab_dict.items()}
        # Special tokens
        self.PAD_TOKEN_ID = self.vocab_dict.get("<pad>", 30)  # Ensure pad_token_id exists
        self.BLANK_TOKEN_ID = self.vocab_dict.get("<blank>", 0)
        self.UNK_TOKEN_ID = self.vocab_dict.get("<unk>", 3)

    def encode(self, text):
        """Convert text to token IDs."""
        text = text.lower().strip()
        token_ids = [self.vocab_dict.get(char, self.UNK_TOKEN_ID) for char in text]
        return numpy.array(token_ids, dtype=numpy.int64)

    def decode(self, token_ids):
        """Convert token IDs back to text."""
        return "".join([self.id_to_token.get(i, "?") for i in token_ids])

    def get_vocab(self):
        """Return the vocabulary dictionary (same as transformers tokenizer)."""
        return self.vocab_dict

    @property
    def pad_token_id(self):
        """Provide pad_token_id like transformers tokenizer."""
        return self.PAD_TOKEN_ID  # Ensures compatibility with get_alignments()


@dataclass
class Segment:
    label: str
    start: int
    end: int

    def __repr__(self):
        return f"{self.label}: [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, idx_to_token_map):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1] == path[i2]:
            i2 += 1
        segments.append(Segment(idx_to_token_map[path[i1]], i1, i2 - 1))
        i1 = i2
    return segments


def time_to_frame(time):
    stride_msec = 20
    frames_per_sec = 1000 / stride_msec
    return int(time * frames_per_sec)


def get_spans(tokens, segments, blank):
    ltr_idx = 0
    tokens_idx = 0
    intervals = []
    start, end = (0, 0)
    for seg_idx, seg in enumerate(segments):
        if tokens_idx == len(tokens):
            assert seg_idx == len(segments) - 1
            assert seg.label == blank
            continue
        cur_token = tokens[tokens_idx].split(" ")
        ltr = cur_token[ltr_idx]
        if seg.label == blank:
            continue
        assert seg.label == ltr, f"{seg.label} != {ltr}"
        if (ltr_idx) == 0:
            start = seg_idx
        if ltr_idx == len(cur_token) - 1:
            ltr_idx = 0
            tokens_idx += 1
            intervals.append((start, seg_idx))
            while tokens_idx < len(tokens) and len(tokens[tokens_idx]) == 0:
                intervals.append((seg_idx, seg_idx))
                tokens_idx += 1
        else:
            ltr_idx += 1
    spans = []
    for idx, (start, end) in enumerate(intervals):
        span = segments[start: end + 1]
        if start > 0:
            prev_seg = segments[start - 1]
            if prev_seg.label == blank:
                pad_start = (
                    prev_seg.start
                    if (idx == 0)
                    else int((prev_seg.start + prev_seg.end) / 2)
                )
                span = [Segment(blank, pad_start, span[0].start)] + span
        if end + 1 < len(segments):
            next_seg = segments[end + 1]
            if next_seg.label == blank:
                pad_end = (
                    next_seg.end
                    if (idx == len(intervals) - 1)
                    else math.floor((next_seg.start + next_seg.end) / 2)
                )
                span = span + [Segment(blank, span[-1].end, pad_end)]
        spans.append(span)
    return spans


def load_audio(audio_file, ret_type='np'):
    waveform, audio_sf = librosa.load(audio_file, sr=SAMPLING_FREQ, mono=True)
    if ret_type == 'np':
        return waveform.astype(numpy.float32)
    else:
        import torch
        return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)


def load_transcript(file_path, dictionary):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lyrics_lines = []
    transcript = []
    for line in lines:
        words = line.strip().lower().split()
        cleaned_words = ["".join([char for char in word if char in dictionary and dictionary[char] != 0]) for word in words]
        cleaned_words = [word for word in cleaned_words if word]
        if cleaned_words:
            transcript.extend(cleaned_words)
            lyrics_lines.append(" ".join(cleaned_words))
    return transcript, lyrics_lines

def align(emission, tokens, device):
    import torch
    import torchaudio.functional as F
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    return F.forced_align(emission, targets, blank=0)

def unflatten(list_, lengths):
    i, ret = 0, []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret

def _postprocess_results(text, spans, waveform, num_frames, sample_rate, scores):
    """Post-processes alignment results to return accurate word timestamps."""
    results = []
    ratio = waveform.size(1) / num_frames
    for i, t in enumerate(text):
        span = spans[i]
        x0, x1 = int(ratio * span[0].start), int(ratio * span[-1].end)
        audio_start_sec = x0 / sample_rate
        audio_end_sec = x1 / sample_rate
        score = scores[span[0].start:span[-1].end].sum()
        results.append({"start": audio_start_sec, "end": audio_end_sec, "text": t, "score": score.item()})
    return results

def get_word_stamps(audio_path, transcript_path, model=None, model_type='MMS_FA'):
    import torch
    import torchaudio
    import torchaudio.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform = load_audio(audio_path, ret_type='torch').to(device)
    if model_type == 'MMS_FA':
        bundle = torchaudio.pipelines.MMS_FA
        dictionary = bundle.get_dict(star=None)
        transcript, lyrics_lines = load_transcript(transcript_path, dictionary)
        model = model or bundle.get_model(with_star=False).to(device)
    elif model_type in ('WAV2VEC2_ASR_BASE_960H', 'WAV2VEC2_ASR_BASE_100H', 'WAV2VEC2_ASR_BASE_10M',
                        'WAV2VEC2_ASR_LARGE_10M', 'WAV2VEC2_ASR_LARGE_100H', 'WAV2VEC2_ASR_LARGE_960H',
                        'WAV2VEC2_ASR_LARGE_LV60K_10M', 'WAV2VEC2_ASR_LARGE_LV60K_100H', 'WAV2VEC2_ASR_LARGE_LV60K_960H',
                        'VOXPOPULI_ASR_BASE_10K_DE', 'VOXPOPULI_ASR_BASE_10K_EN', 'VOXPOPULI_ASR_BASE_10K_ES',
                        'VOXPOPULI_ASR_BASE_10K_FR', 'VOXPOPULI_ASR_BASE_10K_IT', 'HUBERT_ASR_LARGE',
                        'HUBERT_ASR_XLARGE'):
        bundle = getattr(torchaudio.pipelines, model_type)
        labels = bundle.get_labels()
        dictionary = {char.lower(): index for index, char in enumerate(labels)}
        transcript, lyrics_lines = load_transcript(transcript_path, dictionary)
        model = model or bundle.get_model().to(device)
    else:
        raise Exception(f"wrong model type: {model_type}")
    with torch.inference_mode():
        emission, _ = model(waveform)
    tokenized_transcript = [dictionary[c] for word in transcript for c in word if c in dictionary and dictionary[c] != 0]
    aligned_tokens, alignment_scores = align(emission, tokenized_transcript, device)
    token_spans = F.merge_tokens(aligned_tokens[0], alignment_scores[0])
    word_spans = unflatten(token_spans, [len(word) for word in transcript])
    word_timestamps = _postprocess_results(transcript, word_spans, waveform, emission.size(1), bundle.sample_rate, alignment_scores)
    return word_timestamps, model, lyrics_lines

def generate_emissions(session, audio_waveform, window_length=30, context_length=2, batch_size=4):
    # Ensure input is 1D
    assert audio_waveform.ndim == 1, "audio_waveform must be a 1D array"
    # Constants
    context = context_length * SAMPLING_FREQ
    window = window_length * SAMPLING_FREQ
    # Calculate required padding
    extension = math.ceil(audio_waveform.shape[0] / window) * window - audio_waveform.shape[0]
    # Pad the waveform (similar to PyTorch version)
    padded_waveform = numpy.pad(audio_waveform, (context, context + extension), mode="constant")
    # Split into overlapping windows
    num_windows = (padded_waveform.shape[0] - 2 * context) // window
    input_windows = numpy.array([
        padded_waveform[i * window: i * window + window + 2 * context]
        for i in range(num_windows)
    ])
    logger.debug(f"🔍 ONNX Inference: Processing {num_windows} windows, batch size: {batch_size}")
    # Batched Inference
    emissions_list = []
    for i in range(0, num_windows, batch_size):
        input_batch = input_windows[i: i + batch_size]
        # Prepare ONNX input
        inputs = {"input_values": input_batch.astype(numpy.float32)}
        outputs = session.run(["logits"], inputs)
        emissions_list.append(outputs[0])
    # Concatenate results
    emissions = numpy.concatenate(emissions_list, axis=0)  # Shape: (TotalFrames, VocabSize)
    # Remove context frames (match PyTorch behavior)
    emissions = emissions[:, time_to_frame(context_length): -time_to_frame(context_length) + 1, ]
    # Flatten to remove batch dimension
    emissions = emissions.reshape(-1, emissions.shape[-1])
    # Remove extra padding
    if time_to_frame(extension / SAMPLING_FREQ) > 0:
        emissions = emissions[: -time_to_frame(extension / SAMPLING_FREQ), :]
    # Apply log softmax (matching PyTorch post-processing)
    emissions = numpy.log(numpy.exp(emissions) / numpy.sum(numpy.exp(emissions), axis=-1, keepdims=True))
    # Add extra dimension for <star> token (same as PyTorch)
    emissions = numpy.concatenate([emissions, numpy.zeros((emissions.shape[0], 1))], axis=1)
    emissions = emissions.astype(numpy.float32)  # Ensure float32
    logger.debug(f"ONNX Raw Emissions dtype: {emissions.dtype}")  # Should be float32
    # Compute stride
    stride = float(audio_waveform.shape[0] * 1000 / emissions.shape[0] / SAMPLING_FREQ)
    logger.debug(f"✅ ONNX Output Shape: {emissions.shape}, Stride: {stride}ms")
    return emissions, math.ceil(stride)


def forced_align(
        log_probs: numpy.ndarray,
        targets: numpy.ndarray,
        input_lengths: Optional[numpy.ndarray] = None,
        target_lengths: Optional[numpy.ndarray] = None,
        blank: int = 0,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if blank in targets:
        raise ValueError(
            f"targets Tensor shouldn't contain blank index. Found {targets}."
        )
    if blank >= log_probs.shape[-1] or blank < 0:
        raise ValueError("blank must be within [0, log_probs.shape[-1])")
    if numpy.max(targets) >= log_probs.shape[-1] and numpy.min(targets) >= 0:
        raise ValueError("targets values must be within [0, log_probs.shape[-1])")
    assert log_probs.dtype == numpy.float32, "log_probs must be float32"
    from ctc_forced_aligner.ctc_aligner import align_sequences

    paths, scores = align_sequences(log_probs, targets, blank)
    return paths, scores


def get_alignments(
        emissions: numpy.ndarray,  # Expect NumPy array from ONNX model
        tokens: list,
        tokenizer,
):
    assert len(tokens) > 0, "Empty transcript"
    logger.debug(f"len(tokens): {len(tokens)}")  # 574
    logger.debug(f"emissions.shape: {emissions.shape}")  # emissions.shape: (1, 12144, 32)
    # Load tokenizer vocabulary
    dictionary = tokenizer.get_vocab()
    dictionary = {k.lower(): v for k, v in dictionary.items()}
    dictionary["<star>"] = len(dictionary)  # Add special token
    logger.debug(f"Tokenizer vocab size: {len(dictionary)}")  # Debugging
    logger.debug(f"Emissions shape: {emissions.shape}")  # Debugging
    # Convert tokens to token indices
    token_indices = []
    for c in " ".join(tokens).split(" "):
        if c in dictionary:
            token_indices.append(dictionary[c])
        else:
            logger.debug(f"WARNING: Token '{c}' not found in vocabulary")  # Debugging
    # Ensure target indices are valid
    blank_id = dictionary.get("<blank>", tokenizer.pad_token_id)
    targets = numpy.asarray([token_indices], dtype=numpy.int64)
    logger.debug(f"Max target index: {numpy.max(targets)}, Model vocab size: {emissions.shape[-1]}")  # Debugging
    if blank_id in targets:
        raise ValueError(f"targets array should not contain blank index ({blank_id}).")
    if blank_id >= emissions.shape[-1] or blank_id < 0:
        raise ValueError("blank must be within [0, log_probs.shape[-1])")
    if numpy.max(targets) >= emissions.shape[-1] or numpy.min(targets) < 0:
        raise ValueError(f"targets values must be within [0, {emissions.shape[-1]})")
    assert emissions.dtype == numpy.float32, "log_probs must be float32"
    path, scores = forced_align(
        numpy.expand_dims(emissions, axis=0),  # Ensure (1, T, C)
        targets,
        blank=blank_id,
    )
    path = path.squeeze().tolist()
    # Map back to tokens
    idx_to_token_map = {v: k for k, v in dictionary.items()}
    segments = merge_repeats(path, idx_to_token_map)
    return segments, scores, idx_to_token_map[blank_id]


def text_normalize(
        text, iso_code, lower_case=True, remove_numbers=True, remove_brackets=False
):
    config = norm_config.get(iso_code, norm_config["*"])
    for field in [
        "lower_case",
        "punc_set",
        "del_set",
        "mapping",
        "digit_set",
        "unicode_norm",
    ]:
        if field not in config:
            config[field] = norm_config["*"][field]
    text = unicodedata.normalize(config["unicode_norm"], text)
    # Convert to lower case
    if config["lower_case"] and lower_case:
        text = text.lower()
    # brackets
    # always text inside brackets with numbers in them. Usually corresponds to "(Sam 23:17)"
    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
    if remove_brackets:
        text = re.sub(r"\([^\)]*\)", " ", text)
    # Apply mappings
    for old, new in config["mapping"].items():
        text = re.sub(old, new, text)
    # Replace punctutations with space
    punct_pattern = r"[" + config["punc_set"]
    punct_pattern += r"]"
    normalized_text = re.sub(punct_pattern, " ", text)
    # remove characters in delete list
    delete_patten = r"[" + config["del_set"] + r"]"
    normalized_text = re.sub(delete_patten, "", normalized_text)
    if remove_numbers:
        digits_pattern = r"[" + config["digit_set"]
        digits_pattern += r"]+"
        complete_digit_pattern = (
                r"^"
                + digits_pattern
                + r"(?=\s)|(?<=\s)"
                + digits_pattern
                + r"(?=\s)|(?<=\s)"
                + digits_pattern
                + r"$"
        )
        normalized_text = re.sub(complete_digit_pattern, " ", normalized_text)
    if config["rm_diacritics"]:
        from unidecode import unidecode
        normalized_text = unidecode(normalized_text)
    # Remove extra spaces
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
    return normalized_text


# iso codes with specialized rules in uroman
special_isos_uroman = [
    "ara",
    "bel",
    "bul",
    "deu",
    "ell",
    "eng",
    "fas",
    "grc",
    "ell",
    "eng",
    "heb",
    "kaz",
    "kir",
    "lav",
    "lit",
    "mkd",
    "mkd2",
    "oss",
    "pnt",
    "pus",
    "rus",
    "srp",
    "srp2",
    "tur",
    "uig",
    "ukr",
    "yid",
]


def normalize_uroman(text):
    text = text.lower()
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()


def get_uroman_tokens(norm_transcripts, iso=None):
    input_text = "\n".join(norm_transcripts) + "\n"
    from unidecode import unidecode

    tmp = [unidecode(text) for text in norm_transcripts]
    outtexts = []
    for text in tmp:
        # Split characters and add spaces between them
        line = " ".join(text.strip())
        line = re.sub(r"\s+", " ", line).strip()  # Normalize spaces
        outtexts.append(line)
    uromans = [normalize_uroman(ot) for ot in outtexts]
    return uromans


def split_text(text: str, split_size: str = "word"):
    if split_size == "sentence":
        from nltk.tokenize import PunktSentenceTokenizer

        sentence_checker = PunktSentenceTokenizer()
        sentences = sentence_checker.sentences_from_text(text)
        return sentences

    elif split_size == "word":
        return text.split()
    elif split_size == "char":
        return list(text)


def preprocess_text(
        text, romanize, language, split_size="word", star_frequency="segment"
):
    assert split_size in [
        "sentence",
        "word",
        "char",
    ], "Split size must be sentence, word, or char"
    assert star_frequency in [
        "segment",
        "edges",
    ], "Star frequency must be segment or edges"
    if language in ["jpn", "chi"]:
        split_size = "char"
    text_split = split_text(text, split_size)
    norm_text = [text_normalize(line.strip(), language) for line in text_split]

    if romanize:
        tokens = get_uroman_tokens(norm_text, language)
    else:
        tokens = [" ".join(list(word)) for word in norm_text]

    # add <star> token to the tokens and text
    # it's used extensively here but I found that it produces more accurate results
    # and doesn't affect the runtime
    if star_frequency == "segment":

        tokens_starred = []
        [tokens_starred.extend(["<star>", token]) for token in tokens]

        text_starred = []
        [text_starred.extend(["<star>", chunk]) for chunk in text_split]

    elif star_frequency == "edges":
        tokens_starred = ["<star>"] + tokens + ["<star>"]
        text_starred = ["<star>"] + text_split + ["<star>"]

    return tokens_starred, text_starred


def merge_segments(segments, threshold=0.00):
    for i in range(len(segments) - 1):
        if segments[i + 1]["start"] - segments[i]["end"] < threshold:
            segments[i + 1]["start"] = segments[i]["end"]


def postprocess_results(text_starred: list, spans: list, stride: float, scores: numpy.ndarray,
                        merge_threshold: float = 0.0):
    results = []
    for i, t in enumerate(text_starred):
        if t == "<star>":
            continue
        span = spans[i]
        seg_start_idx = span[0].start
        seg_end_idx = span[-1].end

        audio_start_sec = seg_start_idx * (stride) / 1000
        audio_end_sec = seg_end_idx * (stride) / 1000
        score = scores[seg_start_idx:seg_end_idx].sum()
        sample = {
            "start": audio_start_sec,
            "end": audio_end_sec,
            "text": t,
            "score": score.item(),
        }
        results.append(sample)
    merge_segments(results, merge_threshold)
    return results


def ensure_onnx_model(model_path: str, url: str):
    """Downloads the ONNX model file if it does not exist locally."""
    if not os.path.exists(model_path):
        logger.debug(f"Downloading ONNX model from {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.debug(f"Model downloaded successfully to {model_path}.")
        else:
            raise Exception(f"Failed to download the model. HTTP Status Code: {response.status_code}")


def _generate_subtitle(self, input_audio_path, input_text_path, output_st_path, language="eng",
                       batch_size=4, subtitle_type='srt'):
    if self.input_audio_path != input_audio_path or self.input_text_path != input_text_path:
        self.input_audio_path = input_audio_path
        self.input_text_path = input_text_path
        audio_waveform = load_audio(input_audio_path)
        if os.path.exists(input_text_path):
            with open(input_text_path, "r") as f:
                text = f.read().replace("\n", " ").strip()
        else:
            text = input_text_path.replace("\n", " ").strip()
        emissions, stride = generate_emissions(self.model, audio_waveform, batch_size=batch_size)
        tokens_starred, text_starred = preprocess_text(text, romanize=True, language=language)
        segments, scores, blank_token = get_alignments(emissions, tokens_starred, self.tokenizer)
        spans = get_spans(tokens_starred, segments, blank_token)
        self.word_timestamps = postprocess_results(text_starred, spans, stride, scores)
        if os.path.exists(input_text_path):
            with open(input_text_path, "r") as f:
                self.lyrics_lines = [line.strip() for line in f if line.strip()]
        else:
            self.lyrics_lines = input_text_path.split("\n")
    if subtitle_type == 'srt':
        return _generate_srt(self.word_timestamps,  self.lyrics_lines, output_st_path)
    elif subtitle_type == 'webvtt':
        return _generate_webvtt(self.word_timestamps,  self.lyrics_lines, output_st_path)


def _generate_srt(word_timestamps,  lyrics_lines, output_srt_path, lyrics_lines_raw=None):
    # Generate line-level timestamps by mapping words to lines
    line_timestamps = []
    current_line = ""
    current_start = None
    current_end = None
    line_index = 0
    for word in word_timestamps:
        if current_start is None:
            current_start = word["start"]
        current_end = word["end"]
        current_line += word["text"] + " "
        # If the line in the lyrics matches, finalize this segment
        if line_index < len(lyrics_lines) and lyrics_lines[line_index].endswith(current_line.strip()):
            line_timestamps.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "text": lyrics_lines[line_index] if lyrics_lines_raw is None else lyrics_lines_raw[line_index],
                }
            )
            current_line = ""
            current_start = None
            line_index += 1
    # Save the SRT file
    with open(output_srt_path, "w") as f:
        for i, entry in enumerate(line_timestamps, start=1):
            start = entry["start"]
            end = entry["end"]
            text = entry["text"]
            f.write(f"{i}\n")
            x = f"{int(start // 3600):02}:{int(start % 3600 // 60):02}:{start % 60:06.3f}".replace(".", ",")
            y = f"{int(end // 3600):02}:{int(end % 3600 // 60):02}:{end % 60:06.3f}".replace(".", ",")
            line = x + " --> " + y + "\n"
            f.write(line)
            text = text[0].upper() + text[1:]
            f.write(f"{text}\n\n")
    logger.debug(f"SRT file saved to {output_srt_path}")
    return os.path.exists(output_srt_path)


def _generate_webvtt(word_timestamps, lyrics_lines, output_vtt_path, lyrics_lines_raw=None):
    # Generate line-level timestamps
    line_timestamps = []
    current_line = ""
    current_start = None
    line_index = 0
    for word in word_timestamps:
        if current_start is None:
            current_start = word["start"]
        current_end = word["end"]
        current_line += word["text"] + " "
        # If the current line matches a lyrics line, store timestamp
        if line_index < len(lyrics_lines) and lyrics_lines[line_index] == current_line.strip():
            line_timestamps.append({"start": current_start, "end": current_end,
                                    "text": lyrics_lines[line_index] if lyrics_lines_raw is None else lyrics_lines_raw[
                                        line_index],
                                    })
            current_line = ""
            current_start = None
            line_index += 1
    # Save WebVTT file
    with open(output_vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for entry in line_timestamps:
            start = entry["start"]
            end = entry["end"]
            text = entry["text"]
            start_time = f"{int(start // 3600):02}:{int(start % 3600 // 60):02}:{start % 60:06.3f}"
            end_time = f"{int(end // 3600):02}:{int(end % 3600 // 60):02}:{end % 60:06.3f}"
            text = text[0].upper() + text[1:]
            f.write(f"{start_time} --> {end_time}\n{text}\n\n")
    logger.debug(f"WebVTT file saved to {output_vtt_path}")
    return os.path.isfile(output_vtt_path)


class AlignmentSingleton:
    _instance = None

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super(AlignmentSingleton, cls).__new__(cls)
            cls._instance.model_path = kwargs.get("model_path",
                                                  os.path.join(os.path.expanduser("~"), "ctc_forced_aligner",
                                                               "model.onnx"))
            cls._instance._load_model()
        return cls._instance

    @classmethod
    def is_loaded(cls):
        return cls._instance is not None

    def _load_model(self):
        logger.debug(f"Loading ONNX alignment model from '{self.model_path}'...")
        ensure_onnx_model(self.model_path, MODEL_URL)
        self.model = onnxruntime.InferenceSession(self.model_path)  # Load ONNX model
        self.tokenizer = Tokenizer()
        self.word_timestamps = None
        self.lyrics_lines = None
        self.input_audio_path = None
        self.input_text_path = None

    @property
    def alignment_model(self):
        return self.model

    @property
    def alignment_tokenizer(self):
        return self.tokenizer

    def generate_srt(self, input_audio_path, input_text_path, output_srt_path, language="eng", batch_size=4):
        return _generate_subtitle(self, input_audio_path, input_text_path, output_srt_path, language, batch_size,
                                  subtitle_type='srt')

    def generate_webvtt(self, input_audio_path, input_text_path, output_vtt_path, language="eng", batch_size=4):
        return _generate_subtitle(self, input_audio_path, input_text_path, output_vtt_path, language, batch_size,
                                  subtitle_type='webvtt')


class Alignment:
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path)
        self.tokenizer = Tokenizer()

    @property
    def alignment_model(self):
        return self.model

    @property
    def alignment_tokenizer(self):
        return self.tokenizer

    def generate_srt(self, input_audio_path, input_text_path, output_srt_path, language="eng", batch_size=4):
        return _generate_subtitle(self.model, self.tokenizer, input_audio_path, input_text_path, output_srt_path,
                                  language, batch_size)

    def generate_webvtt(self, input_audio_path, input_text_path, output_vtt_path, language="eng", batch_size=4):
        return _generate_webvtt(self.model, self.tokenizer, input_audio_path, input_text_path, output_vtt_path,
                                language, batch_size)


class AlignmentTorchSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AlignmentTorchSingleton, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.input_audio_path = None
        self.input_text_path = None
        self.model_type = None
        self.word_timestamps = None
        self.lyrics_lines = None
        self.lyrics_lines_raw = None
        self.model = None

    @classmethod
    def is_loaded(cls):
        return cls._instance is not None

    def _process_input(self, input_audio_path, input_text_path, model_type):
        if (self.input_audio_path != input_audio_path or
                self.input_text_path != input_text_path or
                self.model_type != model_type):
            self.input_audio_path = input_audio_path
            self.input_text_path = input_text_path
            self.model_type = model_type
            self.word_timestamps, self.model, self.lyrics_lines = get_word_stamps(
                input_audio_path, input_text_path, self.model, model_type
            )

            with open(input_text_path, "r") as f:
                self.lyrics_lines_raw = [line.strip() for line in f if line.strip()]

    def generate_srt(self, input_audio_path, input_text_path, output_srt_path, model_type='MMS_FA'):
        self._process_input(input_audio_path, input_text_path, model_type)
        return _generate_srt(self.word_timestamps, self.lyrics_lines, output_srt_path, self.lyrics_lines_raw)

    def generate_webvtt(self, input_audio_path, input_text_path, output_vtt_path, model_type='MMS_FA'):
        self._process_input(input_audio_path, input_text_path, model_type)
        return _generate_webvtt(self.word_timestamps, self.lyrics_lines, output_vtt_path, self.lyrics_lines_raw)


class AlignmentTorch:
    def __init__(self):
        self.input_audio_path = None
        self.input_text_path = None
        self.model_type = None
        self.word_timestamps = None
        self.lyrics_lines = None
        self.lyrics_lines_raw = None
        self.model = None

    def _process_input(self, input_audio_path, input_text_path, model_type):
        if (self.input_audio_path != input_audio_path or
                self.input_text_path != input_text_path or
                self.model_type != model_type):
            self.input_audio_path = input_audio_path
            self.input_text_path = input_text_path
            self.model_type = model_type
            self.word_timestamps, self.model, self.lyrics_lines = get_word_stamps(
                input_audio_path, input_text_path, self.model, model_type
            )

            with open(input_text_path, "r") as f:
                self.lyrics_lines_raw = [line.strip() for line in f if line.strip()]

    def generate_srt(self, input_audio_path, input_text_path, output_srt_path, model_type='MMS_FA'):
        self._process_input(input_audio_path, input_text_path, model_type)
        return _generate_srt(self.word_timestamps, self.lyrics_lines, output_srt_path, self.lyrics_lines_raw)

    def generate_webvtt(self, input_audio_path, input_text_path, output_vtt_path, model_type='MMS_FA'):
        self._process_input(input_audio_path, input_text_path, model_type)
        return _generate_webvtt(self.word_timestamps, self.lyrics_lines, output_vtt_path, self.lyrics_lines_raw)
