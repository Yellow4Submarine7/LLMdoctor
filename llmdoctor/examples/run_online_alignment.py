import os
import sys
import torch

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llmdoctor.configs.config_loader import load_config
from llmdoctor.models.model_loader import load_model_and_tokenizer
from llmdoctor.core.online_alignment import OnlineAlignment
# For this example, the DoctorModel will be a standard Hugging Face model (mocked here),
# as the OnlineAlignment class is designed to work with models that provide next-token logits.
# The custom DoctorModel from tfpo_tuner.py can also be used if its forward method provides such logits.
# from llmdoctor.core.tfpo_tuner import DoctorModel as TF policíaPODoctorModel # If we wanted to use the custom one

# Mock models and tokenizers for demonstration without heavy loading
class MockHFModelForAlignment:
    def __init__(self, model_name="mock_model", vocab_size=1000, device='cpu'):
        class Config:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
        self.config = Config(vocab_size)
        self.name = model_name
        self.device = device
        print(f"MockHFModelForAlignment '{self.name}' initialized with vocab_size: {vocab_size} on {device}.")

    def eval(self):
        print(f"MockHFModelForAlignment '{self.name}' set to eval mode.")

    def to(self, device):
        self.device = device
        print(f"MockHFModelForAlignment '{self.name}' moved to {device}.")
        return self

    # The OnlineAlignment class expects the model to be callable or have specific methods
    # to get next token logits. For this mock, get_next_token_log_probs_patient/doctor in
    # OnlineAlignment will generate random logits if this model is passed.
    # If this model were to be directly called:
    # def __call__(self, input_ids, attention_mask=None):
    #     batch_size, seq_len = input_ids.shape
    #     logits = torch.randn(batch_size, seq_len, self.config.vocab_size).to(self.device)
    #     return {'logits': logits}


class MockHFTokenizerForAlignment:
    def __init__(self, model_name="mock_tokenizer", vocab_size=1000, eos_token_id=None):
        self.name = model_name
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id if eos_token_id is not None else vocab_size -1 # Default EOS
        self.pad_token_id = self.eos_token_id # Common practice
        print(f"MockHFTokenizerForAlignment '{self.name}' initialized with vocab_size: {vocab_size}, EOS: {self.eos_token_id}.")

    def encode(self, text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=None):
        # print(f"Tokenizer '{self.name}': Encoding text: '{text}'")
        tokens = [min(ord(c) % self.vocab_size, self.vocab_size-1) for c in text if ord(c) < self.vocab_size]
        if not tokens: tokens = [0] # Handle empty
        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long).to('cpu') # Keep on CPU, models will move
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        # print(f"Tokenizer '{self.name}': Decoding IDs: {token_ids}")
        chars = []
        # Handle cases where token_ids might be a list of tensors or a single tensor of IDs
        if isinstance(token_ids, torch.Tensor):
            token_ids_list = token_ids.tolist()
        else: # Assuming list of items (potentially tensors)
            token_ids_list = [tid.item() if isinstance(tid, torch.Tensor) else tid for tid in token_ids]

        for tid in token_ids_list:
            if skip_special_tokens and tid == self.eos_token_id:
                continue
            chars.append(chr(97 + (tid % 26))) # Map to a-z
        return "".join(chars) if chars else "empty_decode"

def main():
    print("--- Running LLMdoctor: Stage 3 - Online Alignment with Flow-Guided Reward Model ---")

    # 1. Load Configuration
    print("\n1. Loading configuration...")
    config_path = os.path.join(project_root, 'llmdoctor', 'configs', 'default_config.yaml')
    if not os.path.exists(config_path):
        print(f"Warning: Default config not found at {config_path}. Using load_config's internal default.")
        config = load_config()
    else:
        config = load_config(config_path)
        
    if not config:
        print("Configuration could not be loaded. Exiting.")
        return

    online_align_params = config.get('online_alignment_params', {})
    training_settings = config.get('training_settings', {})
    model_paths_config = config.get('model_paths', {})

    device = torch.device("cuda" if torch.cuda.is_available() and training_settings.get('device') != 'cpu' else "cpu")
    if training_settings.get('device') == 'auto':
        print(f"Device set to 'auto'. Using: {device}")
    else:
        print(f"Device set to: {device} (based on config and availability)")

    torch.manual_seed(training_settings.get('seed', 42))
    print(f"Set random seed to: {training_settings.get('seed', 42)}")

    # 2. Load Patient Model and Tokenizer (Mocks for this script)
    print("\n2. Loading Patient Model and Tokenizer (using Mocks)...")
    patient_model_name = model_paths_config.get('patient_model_name', "mock_patient")
    
    patient_vocab_size = 3000 # Larger mock vocab for patient
    patient_model = MockHFModelForAlignment(model_name="PatientLM", vocab_size=patient_vocab_size, device=device)
    patient_tokenizer = MockHFTokenizerForAlignment(model_name="PatientTokenizer", vocab_size=patient_vocab_size, eos_token_id=patient_vocab_size-1)
    print("Mock Patient Model and Tokenizer loaded.")

    # 3. Load Doctor Model(s) and Tokenizer (Mocks for this script)
    print("\n3. Loading Doctor Model(s) and Tokenizer (using Mocks)...")
    doctor_model_name = model_paths_config.get('doctor_model_name', "mock_doctor")
    
    doctor_vocab_size = patient_vocab_size 
    doctor_model_1 = MockHFModelForAlignment(model_name="DoctorLM_1", vocab_size=doctor_vocab_size, device=device)
    
    doctor_tokenizer = MockHFTokenizerForAlignment(model_name="DoctorTokenizer", vocab_size=doctor_vocab_size, eos_token_id=doctor_vocab_size-1)
    
    doctor_models_list = [doctor_model_1] 
    print(f"Mock Doctor Model(s) ({len(doctor_models_list)}) and Tokenizer loaded.")


    # 4. Initialize OnlineAlignment
    print("\n4. Initializing OnlineAlignment module...")
    online_aligner = OnlineAlignment(
        patient_model=patient_model,
        doctor_models=doctor_models_list, 
        patient_tokenizer=patient_tokenizer,
        doctor_tokenizer=doctor_tokenizer 
    )
    print("OnlineAlignment module initialized.")

    # 5. Run Guided Generation
    print("\n5. Starting guided generation...")
    prompts = [
        "Explain the theory of relativity in simple terms:",
        "What is the capital of France?",
        "Write a short poem about spring:"
    ]

    alpha = online_align_params.get('alpha', 1.0)
    beta_values = online_align_params.get('beta_values', [0.5] * len(doctor_models_list))
    if len(beta_values) != len(doctor_models_list):
        print(f"Warning: Number of beta_values ({len(beta_values)}) in config does not match number of doctor models ({len(doctor_models_list)}). Using default betas.")
        beta_values = [0.5] * len(doctor_models_list)


    for i, prompt_text in enumerate(prompts):
        print(f"\n--- Generating for Prompt {i+1}/{len(prompts)} ---")
        print(f"Prompt: {prompt_text}")
        
        generated_text = online_aligner.generate_sequence(
            prompt_text=prompt_text,
            max_length=online_align_params.get('max_generation_length', 50),
            alpha=alpha,
            beta_values=beta_values, 
            temperature=online_align_params.get('temperature', 0.7),
            top_k=online_align_params.get('top_k', 50)
        )
        print(f"Generated Text: {generated_text}")

    print("\n--- Online Alignment Stage Complete ---")
    print("Note: This script used mock models. Generated text is random.")
    print("To generate meaningful aligned text, use actual trained patient and doctor models.")

if __name__ == "__main__":
    main()
