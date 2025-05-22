from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Default model names (can be overridden by config)
DEFAULT_PATIENT_MODEL_NAME = "Qwen/Qwen1.5-72B-Chat" # Example large model
DEFAULT_DOCTOR_MODEL_NAME = "Qwen/Qwen1.5-4B-Chat"   # Example smaller model

def load_model_and_tokenizer(model_name_or_path: str, 
                             use_quantization: bool = False, 
                             low_cpu_mem_usage: bool = True,
                             is_doctor_model: bool = False):
    """
    Loads a Hugging Face model and its tokenizer.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model.
        use_quantization (bool): Whether to use 4-bit quantization (BNB).
                                 Helpful for loading large models with less VRAM.
        low_cpu_mem_usage (bool): Enables loading model shards subsequently to reduce CPU RAM usage.
                                  Requires `accelerate` to be installed.
        is_doctor_model (bool): If true, loads the model with potentially fewer optimizations,
                                assuming it's smaller.

    Returns:
        tuple: (model, tokenizer)
               The loaded Hugging Face model and tokenizer.
               Returns (None, None) if loading fails.
    """
    print(f"Loading model and tokenizer for: {model_name_or_path}")
    
    quantization_config = None
    if use_quantization:
        print("Using 4-bit quantization (BitsAndBytesConfig).")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 # or torch.float16
        )

    try:
        # Load tokenizer
        # `trust_remote_code=True` might be needed for some models like Qwen
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        print(f"Tokenizer for {model_name_or_path} loaded successfully.")

        # Set pad_token to eos_token if not already set (common for many causal LMs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": low_cpu_mem_usage if not is_doctor_model else False, # Typically for very large models
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto" # device_map='auto' is recommended for BNB quantization
        else:
            # If not quantizing, device_map can be omitted or set to a specific device if desired
            # For smaller doctor models, explicit device mapping might not be needed if loaded on CPU first
            pass


        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        print(f"Model {model_name_or_path} loaded successfully.")
        
        # For non-quantized models, ensure they are on the correct device (e.g. GPU if available)
        # If using device_map="auto" with quantization, this is handled.
        # If not using quantization, and no device_map, model loads on CPU by default.
        # We might move it to a device later if needed.
        
        # Set model to evaluation mode by default
        model.eval()
        print(f"Model {model_name_or_path} set to evaluation mode.")

        return model, tokenizer

    except Exception as e:
        print(f"Error loading model or tokenizer for {model_name_or_path}: {e}")
        return None, None

# Example: Wrapper classes (Optional, but can help standardize interfaces)
# For now, the core logic in online_alignment.py and tfpo_tuner.py will use
# the Hugging Face model objects directly or the custom DoctorModel nn.Module.
# If wrappers become necessary, they can be defined here.

# class PatientModelWrapper:
#     def __init__(self, model, tokenizer):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.device = model.device if hasattr(model, 'device') else torch.device("cpu")

#     def get_log_probs(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None):
#         outputs = self.model(input_ids=token_ids, attention_mask=attention_mask)
#         logits = outputs.logits[:, -1, :]
#         return torch.log_softmax(logits, dim=-1)

# class DoctorModelWrapper: # If wrapping a HF model used as a doctor
#     def __init__(self, model, tokenizer):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.device = model.device if hasattr(model, 'device') else torch.device("cpu")

#     def get_preference_scores(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None):
#         # This would be equivalent to the forward pass of the TFPO DoctorModel
#         outputs = self.model(input_ids=token_ids, attention_mask=attention_mask)
#         logits = outputs.logits[:, -1, :] # Logits for all next tokens
#         return torch.log_softmax(logits, dim=-1) # Or raw logits if preferred by alignment algorithm


if __name__ == '__main__':
    print("--- Testing Model Loader ---")
    
    # Test loading a smaller model (e.g., a very small GPT-2 for quick testing if Qwen models are too large)
    # Replace with a small, fast-loading model you have access to or is easily downloadable.
    # For example: "sshleifer/tiny-gpt2" or "prajjwal1/bert-tiny" (though BERT is not causal)
    # Using a very small dummy model name that might not exist to show error handling:
    # test_model_name = "dummy-nonexistent-model" 
    # print(f"\nAttempting to load (potentially failing): {test_model_name}")
    # model, tokenizer = load_model_and_tokenizer(test_model_name, use_quantization=False)
    # if model and tokenizer:
    #     print(f"Successfully loaded {test_model_name}")
    # else:
    #     print(f"Failed to load {test_model_name}, as expected for a dummy name or if model is unavailable.")

    # Example of loading a small, real model like "sshleifer/tiny-gpt2"
    # This model is very small and good for testing the loading mechanism without large downloads.
    small_test_model_name = "sshleifer/tiny-gpt2" # (1.5M parameters)
    print(f"\nAttempting to load small test model: {small_test_model_name}")
    # Not using quantization for such a tiny model.
    model_sm, tokenizer_sm = load_model_and_tokenizer(small_test_model_name, use_quantization=False, low_cpu_mem_usage=False)
    if model_sm and tokenizer_sm:
        print(f"Successfully loaded {small_test_model_name}.")
        print(f"Model config: {model_sm.config.model_type}, Vocab size: {model_sm.config.vocab_size}")
        
        # Test basic tokenization and model usage (if it's a causal LM)
        try:
            prompt = "Hello, world!"
            inputs = tokenizer_sm(prompt, return_tensors="pt")
            print(f"Tokenized prompt '{prompt}': {inputs['input_ids']}")
            if hasattr(model_sm, 'generate'): # Check if it's a generative model
                 outputs = model_sm.generate(inputs['input_ids'], max_length=10)
                 print(f"Generated output (ids): {outputs}")
                 print(f"Generated text: {tokenizer_sm.decode(outputs[0])}")
            else: # If not generative, just try a forward pass
                outputs = model_sm(**inputs)
                print(f"Model output keys (if forward pass): {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")

        except Exception as e:
            print(f"Error during basic usage test for {small_test_model_name}: {e}")
    else:
        print(f"Failed to load {small_test_model_name}. Ensure it's a valid Hugging Face model name.")

    print("\n--- Model Loader Test Complete ---")
    print("Note: To test with larger models like Qwen, ensure you have sufficient resources and have accepted Hugging Face terms if necessary.")
    print("Loading large models can take a significant amount of time and resources.")
    print("Consider using `use_quantization=True` for large models to save memory.")
