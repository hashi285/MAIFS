import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
MAX_MEMORY = {0: "30GiB", 1: "30GiB", 2: "30GiB", 3: "30GiB"}
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TURNS = 8


def main() -> None:
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=MAX_MEMORY,
        trust_remote_code=True,
    )
    model.eval()

    print("Ready. Type 'exit' or 'quit' to stop.")
    history = []

    while True:
        user_text = input("\nuser> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": user_text})
        if len(history) > MAX_TURNS * 2:
            history = history[-MAX_TURNS * 2 :]

        inputs = tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        target_device = getattr(model, "device", "cuda:0")
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": text})
        print(f"assistant> {text}")


if __name__ == "__main__":
    main()
