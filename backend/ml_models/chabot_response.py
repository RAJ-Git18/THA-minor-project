from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import AutoConfig


def get_model_response(user_input):

    model_name = "Pranilllllll/finetuned_gpt2_45krows_10epochs"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    text_generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer)

    while True:
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break

        response = text_generator(
            user_input,
            max_length=500,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            truncation=False,
        )

        generated_text = response[0]["generated_text"]

        if generated_text.startswith(user_input):
            suggestion = generated_text[len(user_input) :].strip()
        else:
            suggestion = generated_text.strip()

        print(f"Bot: {suggestion}")
        return suggestion
