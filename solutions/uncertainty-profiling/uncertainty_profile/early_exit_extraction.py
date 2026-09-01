"""Single-pass prefix feature extraction for early robustness detection."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.logits_process import LogitsProcessorList

from uncertainty_profile.config import GenerationConfidenceConfig
from uncertainty_profile.early_exit import normalize_budgets, prefix_name
from uncertainty_profile.extraction import (
    FeatureRow,
    GenerationConfidenceCollector,
    _token_id_set,
    build_valid_token_mask,
    format_problem,
    get_model_input_device,
)
from uncertainty_profile.metrics import (
    ConfidenceMetrics,
    compute_generation_confidence_metrics,
)
from uncertainty_profile.temporal_metrics import (
    TREND_FEATURE_NAMES,
    compute_temporal_uncertainty_metrics,
)


class EarlyExitGenerationConfidenceCollector(GenerationConfidenceCollector):
    """Compute 14D+12D features for several token prefixes in one generation."""

    def __init__(self, budgets: Sequence[int]) -> None:
        super().__init__()
        self.budgets = normalize_budgets(budgets)

    def compute_features(
        self,
        *,
        sequences: torch.Tensor,
        prompt_length: int,
        pad_token_id: int | None,
        eos_token_ids: set[int],
        config: GenerationConfidenceConfig,
    ) -> tuple[list[ConfidenceMetrics], torch.Tensor]:
        """Return ordinary full-trace metrics plus namespaced prefix features."""

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
        top1_token_ids = torch.stack(
            self.top1_token_ids, dim=1
        ).detach().cpu()
        selected_is_top1 = generated_token_ids == top1_token_ids

        valid_mask = build_valid_token_mask(
            generated_token_ids,
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
        )

        for batch_index, row in enumerate(rows):
            mask = valid_mask[batch_index]

            def masked_numpy(values: torch.Tensor) -> np.ndarray:
                return values[batch_index][mask].numpy()

            log_probs = masked_numpy(selected_log_probs)
            probs = masked_numpy(selected_probs)
            entropy_trace = masked_numpy(entropy)
            top1_trace = masked_numpy(top1_probs)
            margin_trace = masked_numpy(top2_margins)
            top1_selected = masked_numpy(selected_is_top1)

            for budget in self.budgets:
                prefix_len = min(int(budget), int(log_probs.size))
                prefix_slice = slice(0, prefix_len)

                official = compute_generation_confidence_metrics(
                    log_probs=log_probs[prefix_slice],
                    probs=probs[prefix_slice],
                    entropy=entropy_trace[prefix_slice],
                    top1_probs=top1_trace[prefix_slice],
                    top2_margins=margin_trace[prefix_slice],
                    selected_is_top1=top1_selected[prefix_slice],
                    min_k_fraction=config.min_k_fraction,
                    high_conf_threshold=config.high_conf_threshold,
                    low_conf_threshold=config.low_conf_threshold,
                )
                row[prefix_name(budget, "num_tokens")] = int(
                    official["generation_num_tokens"]
                )
                for feature_name, value in official.items():
                    if feature_name == "generation_num_tokens":
                        continue
                    row[prefix_name(budget, feature_name)] = value

                temporal = compute_temporal_uncertainty_metrics(
                    log_probs=log_probs[prefix_slice],
                    probs=probs[prefix_slice],
                    entropy=entropy_trace[prefix_slice],
                    top1_probs=top1_trace[prefix_slice],
                    top2_margins=margin_trace[prefix_slice],
                    high_conf_threshold=config.high_conf_threshold,
                    low_conf_threshold=config.low_conf_threshold,
                )
                for feature_name in TREND_FEATURE_NAMES:
                    row[prefix_name(budget, feature_name)] = temporal[feature_name]

        return rows, generated_token_ids


@torch.inference_mode()
def iter_early_exit_generation_feature_batches(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problem_texts: Sequence[str],
    config: GenerationConfidenceConfig,
    budgets: Sequence[int],
    include_generated_text: bool = False,
) -> Iterator[tuple[int, list[FeatureRow]]]:
    """Generate once to max budget and summarize every requested prefix."""

    config.validate()
    budgets = normalize_budgets(budgets)
    if config.max_new_tokens < max(budgets):
        raise ValueError(
            "max_new_tokens must be at least the largest early-exit budget; "
            f"got {config.max_new_tokens} < {max(budgets)}"
        )

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

        collector = EarlyExitGenerationConfidenceCollector(budgets)
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
