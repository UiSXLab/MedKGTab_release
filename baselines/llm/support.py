import json

import numpy as np
import pandas as pd


SYSTEM = """\
You are a synthetic data generator tasked with completing source-only samples into full samples.

1. Analyze the provided real samples carefully. Each real sample is a JSON object containing BOTH source fields and target fields.
2. When given source-only samples, infer and generate the missing target fields based on patterns learned from the real samples.
3. Maintain realistic relationships/correlations between source and target fields as reflected in the real samples.

IMPORTANT (must-follow):
- For each source-only sample, you MUST copy the source fields and their values EXACTLY as provided.
  Do NOT change, normalize, round, reorder, rename, or regenerate source values.
- ONLY generate the missing target fields so that the completed sample has the SAME set of keys as real samples.
- Do NOT use a trivial completion such as filling all target fields with 0 or the same constant template,
  unless this is strongly supported by very similar real samples.
- Target values must reflect the real-sample distributions: use realistic ranges, non-zero rates, and variability seen in real samples.
  If a target field is frequently non-zero in real samples, it should not be set to 0 by default.

Output (STRICT):
- Output ONLY a valid JSON array, with length equal to the number of source-only input lines, in the same order.
- Do NOT wrap in Markdown/code fences.
- Do NOT output any extra text before or after the JSON.
"""

USER = """\
Missing target fields: {target_fields}

Real Samples (each line is one JSON object):
<<<REAL_SAMPLES_START>>>
{real_samples}
<<<REAL_SAMPLES_END>>>

Source-only Samples to Complete (each line is one JSON object):
<<<SOURCE_ONLY_START>>>
{partial_samples}
<<<SOURCE_ONLY_END>>>
"""


def records(frame):
    return "\n".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False)
        for record in frame.to_dict(orient="records")
    )


def messages(real, query):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER.format(
            target_fields=json.dumps([name for name in real.columns if name not in query.columns]),
            real_samples=records(real), partial_samples=records(query)
        )},
    ]


def complete(client, model, real, query, targets, *, temperature, max_tokens):
    """Parse original array responses; keep observed source values from input."""
    prompt = messages(real, query)
    for attempt in range(3):
        response = client.chat.completions.create(
            model=model, messages=prompt, temperature=temperature, max_tokens=max_tokens
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ValueError("Response truncated; increase --max-tokens.")
        content = (choice.message.content or "").strip()
        try:
            if content.startswith("```") and content.endswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                from json_repair import repair_json
                result = json.loads(repair_json(content))
            if isinstance(result, dict) and len(query) == 1:
                result = [result]
            if not isinstance(result, list) or len(result) != len(query):
                raise ValueError("The response must contain one record per query.")
            if any(not isinstance(row, dict) for row in result):
                raise ValueError("Each completed record must be an object.")
            if any(target not in row for row in result for target in targets):
                raise ValueError("Generated records are missing target columns.")
            values = [[row[target] for target in targets] for row in result]
            if any(type(value) not in (float, int) for row in values for value in row):
                raise ValueError("Target values must be finite JSON numbers.")
            values = np.asarray(values, dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Target values must be finite JSON numbers.")
            generated = pd.DataFrame(values, columns=targets)
            return pd.concat([query.reset_index(drop=True), generated], axis=1)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            if attempt == 2:
                raise ValueError("The API returned invalid predictions three times.") from error
