"""Temporal feature extraction built on the official online confidence collector."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.logits_process import LogitsProcessorList

from uncertainty_profile.config import GenerationConfidenceConfig
from uncertainty_profile.extraction import (
    FeatureRow,
    GenerationConfidenceCollector,
    _token_id_set,
    build_valid_token_mask,
    format_problem,
    get_model_input_device,
)
from uncertainty_profile.metrics import ConfidenceMetrics
from uncertainty_profile.temporal_metrics import (
    compute_temporal_uncertainty_metrics,
)


class TemporalGenerationConfidenceCollector(GenerationConfidenceCollector):
    """Extend the official collector with temporal summaries of its traces."""

    def compute_features(
        self,
        *,
        sequences: torch.Tensor,
        prompt_length: int,
        pad_token_id: int | None,
        eos_token_ids: set[int],
        config: GenerationConfidenceConfig,
    ) -> tuple[list[ConfidenceMetrics], torch.Tensor]:
        """Return the official metrics plus 52 temporal uncertainty features."""

        rows, generated_token_ids = super().compute_features(
            sequences=sequences,
            prompt_length=prompt_length,
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
            config=config,
        )

        selected_log_probs = torch.stack(
            self.selected_log_probs, dim=1
        ).detach().float().cpu()
        selected_probs = selected_log_probs.exp()
        entropy = torch.stack(self.entropy, dim=1).detach().float().cpu()
        top1_probs = torch.stack(self.top1_probs, dim=1).detach().float().cpu()
        top2_margins = torch.stack(
            self.top2_margins, dim=1
        ).detach().float().cpu()

        valid_mask = build_valid_token_mask(
            generated_token_ids,
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
        )

        for batch_index, row in enumerate(rows):
            mask = valid_mask[batch_index]

            def masked_numpy(values: torch.Tensor) -> np.ndarray:
                return values[batch_index][mask].numpy()

            row.update(
                compute_temporal_uncertainty_metrics(
                    log_probs=masked_numpy(selected_log_probs),
                    probs=masked_numpy(selected_probs),
                    entropy=masked_numpy(entropy),
                    top1_probs=masked_numpy(top1_probs),
                    top2_margins=masked_numpy(top2_margins),
                    high_conf_threshold=config.high_conf_threshold,
                    low_conf_threshold=config.low_conf_threshold,
                )
            )

        return rows, generated_token_ids


@torch.inference_mode()
def iter_temporal_generation_feature_batches(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problem_texts: Sequence[str],
    config: GenerationConfidenceConfig,
    include_generated_text: bool = False,
) -> Iterator[tuple[int, list[FeatureRow]]]:
    """Yield feature batches using the temporal collector."""

    config.validate()
    device = get_model_input_device(model)
    eos_value = getattr(model.generation_config, "eos_token_id", None)
    if eos_value is None:
        eos_value = tokenizer.eos_token_id
    eos_token_ids = _token_id_set(eos_value) | _token_id_set(tokenizer.eos_token_id)

    for start in range(0, len(problem_texts), config.batch_size):
        batch_texts = list(problem_texts[start : start + config.batch_size])
        formatted = [
            format_problem(tokenizer, text, role=config.chat_template_role)
            for text in batch_texts
        ]
        encoded = tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=config.max_prompt_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        prompt_length = int(encoded["input_ids"].shape[1])

        collector = TemporalGenerationConfidenceCollector()
        outputs = model.generate(
            **encoded,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            return_dict_in_generate=True,
            output_scores=False,
            logits_processor=LogitsProcessorList([collector]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_value,
            use_cache=True,
            cache_implementation=config.cache_implementation,
        )
        metric_rows, generated_token_ids = collector.compute_features(
            sequences=outputs.sequences,
            prompt_length=prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_ids=eos_token_ids,
            config=config,
        )

        rows: list[FeatureRow] = [dict(row) for row in metric_rows]
        if include_generated_text:
            generated_texts = tokenizer.batch_decode(
                generated_token_ids,
                skip_special_tokens=True,
            )
            for row, generated_text in zip(rows, generated_texts):
                row["generated_text"] = generated_text

        yield start, rows

        del encoded, outputs, collector, generated_token_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_temporal_generation_features(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problem_texts: Sequence[str],
    config: GenerationConfidenceConfig,
    include_generated_text: bool = False,
) -> list[FeatureRow]:
    """Return temporal feature rows in the same order as the input prompts."""

    rows: list[FeatureRow] = []
    for _, batch_rows in iter_temporal_generation_feature_batches(
        model=model,
        tokenizer=tokenizer,
        problem_texts=problem_texts,
        config=config,
        include_generated_text=include_generated_text,
    ):
        rows.extend(batch_rows)

    if len(rows) != len(problem_texts):
        raise RuntimeError(
            f"generated {len(rows)} feature rows for {len(problem_texts)} prompts"
        )
    return rows
