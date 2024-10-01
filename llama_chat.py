import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import pdb

transformers.logging.set_verbosity_error()

def chat_with_llm(model, tokenizer, sys_prompt=None):
    # Chat loop
    messages = []

    # Fetch user and assistant tokens from the tokenizer
    # Determine user and assistant tokens or set defaults if they are not available
    user_token = "<|start_header_id|>user<|end_header_id|>"
    assistant_token = "<|start_header_id|>assistant<|end_header_id|>" # Use default if attribute not found
    end_of_text_token = tokenizer.eos_token

    if sys_prompt is not None:
        messages.append({"role": "system", "content": sys_prompt})

    while True:
        # Print the input format for transparency
        prompt = input(f"{user_token}\n\n")

        # if user types 'quit' then reset the conversation
        if prompt.lower() == "quit":
            messages = []
            print("Conversation reset.")
            if sys_prompt is not None:
                messages.append({"role": "system", "content": sys_prompt})
            continue

        # Add the user's input to the message list
        messages.append({"role": "user", "content": prompt})

        # Print the formatted user input for full transparency
        print(f"{end_of_text_token}", end="")

        # Apply chat template to format input for the model
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt", device=model.device)

        # Move tokenized input to the appropriate device
        inputs = tokenizer(tokenized_chat, return_tensors="pt", padding=True, truncation=True).to(device)
        # Generate the model's response

        outputs = model.generate(**inputs, max_length=4000, num_return_sequences=1, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.pad_token_id)

        # Decode the model's response
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Add the assistant's response to the messages list
        messages.append({"role": "assistant", "content": response})

        # Print the formatted assistant's response for full transparency
        print(f"{assistant_token}\n\n{response}{end_of_text_token}", end="")

# Set up the model
device = "cpu"

model_name = 'meta-llama/Llama-3.2-1B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)

#pdb.set_trace()

# print num params
print(f"Number of parameters: {model.num_parameters()}")

# add padding token
tokenizer.pad_token = tokenizer.eos_token

# Generate text

chat_with_llm(model, tokenizer)