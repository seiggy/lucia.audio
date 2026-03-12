"""Streaming synthesis for Chatterbox Turbo.

Yields audio chunks during generation by chunking the T3 autoregressive loop
and running incremental S3Gen decode (CFM flow → HiFi-GAN) on each chunk.
Watermarking is skipped for streaming (requires full waveform).
"""

import logging
from typing import AsyncGenerator, Generator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

_LOGGER = logging.getLogger(__name__)

# How many tokens to accumulate before decoding a chunk.
# Turbo tokens map to ~2 mel frames each. At 24kHz with 256-sample hop,
# each mel frame ≈ 10.7ms. So 25 tokens ≈ 535ms of audio.
DEFAULT_CHUNK_TOKENS = 25


def generate_streaming(
    model,
    text: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    temperature: float = 0.8,
    top_k: int = 1000,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    max_gen_len: int = 1000,
) -> Generator[Tuple[np.ndarray, int], None, None]:
    """Streaming synthesis for Chatterbox Turbo.

    Yields (audio_chunk_float32, sample_rate) tuples as audio becomes available.
    Each chunk is a numpy float32 array of raw PCM samples.

    Args:
        model: ChatterboxTurboTTS instance (must have .conds set)
        text: Text to synthesize
        chunk_tokens: Number of speech tokens to accumulate before decoding
        temperature, top_k, top_p, repetition_penalty: Sampling parameters
        max_gen_len: Maximum generation length in tokens
    """
    from chatterbox.tts_turbo import punc_norm
    from chatterbox.models.s3gen.const import S3GEN_SIL

    assert model.conds is not None, "Voice conditionals must be set before streaming"

    device = model.device
    sr = model.sr

    # Normalize and tokenize text
    text = punc_norm(text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(device)

    t3 = model.t3
    s3gen = model.s3gen

    # Build logits processors
    logits_processors = LogitsProcessorList()
    if temperature > 0 and temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        logits_processors.append(TopKLogitsWarper(top_k))
    if top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p))
    if repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

    # Initial prefill
    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = t3.prepare_input_embeds(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )

    llm_outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
    hidden_states = llm_outputs[0]
    past_key_values = llm_outputs.past_key_values

    speech_logits = t3.speech_head(hidden_states[:, -1:])
    processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
    probs = F.softmax(processed_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    all_generated = [next_token]
    chunk_tokens_buf = [next_token.squeeze()]
    current_token = next_token

    # Autoregressive loop with chunked decode
    for step in range(max_gen_len):
        current_embed = t3.speech_emb(current_token)

        llm_outputs = t3.tfmr(
            inputs_embeds=current_embed,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values
        speech_logits = t3.speech_head(hidden_states)

        input_ids = torch.cat(all_generated, dim=1)
        processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])
        if torch.all(processed_logits == -float("inf")):
            break

        probs = F.softmax(processed_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Check EOS
        if torch.all(next_token == t3.hp.stop_speech_token):
            break

        all_generated.append(next_token)
        chunk_tokens_buf.append(next_token.squeeze())
        current_token = next_token

        # Decode chunk when buffer is full
        if len(chunk_tokens_buf) >= chunk_tokens:
            chunk_audio = _decode_token_chunk(s3gen, chunk_tokens_buf, model.conds.gen, device)
            if chunk_audio is not None:
                yield chunk_audio, sr
            chunk_tokens_buf = []

    # Decode remaining tokens + silence padding
    silence = [torch.tensor(S3GEN_SIL, device=device) for _ in range(3)]
    chunk_tokens_buf.extend(silence)
    if chunk_tokens_buf:
        chunk_audio = _decode_token_chunk(s3gen, chunk_tokens_buf, model.conds.gen, device, finalize=True)
        if chunk_audio is not None:
            yield chunk_audio, sr


def _decode_token_chunk(
    s3gen,
    token_list: list,
    ref_dict: dict,
    device: torch.device,
    finalize: bool = False,
) -> Optional[np.ndarray]:
    """Decode a chunk of speech tokens to audio using S3Gen."""
    if not token_list:
        return None

    # Stack tokens, filter invalid
    tokens = torch.stack(token_list).to(device)
    tokens = tokens[tokens < 6561]
    if len(tokens) == 0:
        return None

    tokens = tokens.unsqueeze(0)  # (1, T)

    try:
        # Use the separated flow + hifigan path for chunk decode
        output_mels = s3gen.flow_inference(
            speech_tokens=tokens,
            ref_dict=ref_dict,
            n_cfm_timesteps=2,
            finalize=finalize,
        )
        output_mels = output_mels.to(dtype=s3gen.dtype)
        output_wavs, _ = s3gen.hift_inference(output_mels)
        audio = output_wavs.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return audio
    except Exception as e:
        _LOGGER.warning("Chunk decode failed: %s", e)
        return None
