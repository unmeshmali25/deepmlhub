import torch


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
) -> str:
    """Generate one response for a user prompt."""
    messages = [{"role": "user", "content": prompt}]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    ).to("cpu")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()
