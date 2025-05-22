import torch
import torch.nn.functional as F

# Assuming DoctorModel and PatientModel will be properly imported when integrated
# from ..models.model_loader import PatientModel # Example import
# from .tfpo_tuner import DoctorModel # Example import

class OnlineAlignment:
    def __init__(self, patient_model, doctor_models, patient_tokenizer, doctor_tokenizer):
        """
        Manages the online alignment process.

        Args:
            patient_model: The large "patient" LLM (e.g., loaded Qwen 70B).
            doctor_models (DoctorModel | list[DoctorModel]): A single trained "doctor" model 
                                                             or a list of doctor models for multi-dimensional alignment.
            patient_tokenizer: Tokenizer for the patient model.
            doctor_tokenizer: Tokenizer for the doctor model(s). 
                              (Note: May need to handle cases where tokenizers differ significantly)
        """
        self.patient_model = patient_model
        if not isinstance(doctor_models, list):
            self.doctor_models = [doctor_models]
        else:
            self.doctor_models = doctor_models
        
        self.patient_tokenizer = patient_tokenizer
        self.doctor_tokenizer = doctor_tokenizer # Assuming it's the same for all doctor models if multiple

        # Placeholder: Check if models are on the same device, move if necessary
        # self.device = patient_model.device 
        # for dm in self.doctor_models:
        #    dm.to(self.device)
        print("OnlineAlignment initialized.")
        if len(self.doctor_models) > 1:
            print(f"Initialized with {len(self.doctor_models)} doctor models for multi-dimensional alignment.")

    def get_next_token_log_probs_patient(self, current_token_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Gets the log-probabilities for the next token from the patient model.

        Args:
            current_token_ids (torch.Tensor): Tensor of token IDs generated so far. Shape (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Attention mask for current_token_ids.

        Returns:
            torch.Tensor: Log-probabilities for all tokens in the vocab. Shape (batch_size, vocab_size).
        """
        # This is a placeholder for actual patient model inference
        print(f"Patient model: Getting log_probs for current_token_ids shape {current_token_ids.shape}")
        # Example: outputs = self.patient_model(input_ids=current_token_ids, attention_mask=attention_mask)
        # logits = outputs.logits[:, -1, :] # Get logits for the last token prediction
        # For placeholder:
        vocab_size = self.patient_model.config.vocab_size if hasattr(self.patient_model, 'config') and hasattr(self.patient_model.config, 'vocab_size') else 32000 # fallback vocab size
        
        # Simulate some dependency on input, batch_size aware
        batch_size = current_token_ids.shape[0]
        logits = torch.randn(batch_size, vocab_size).to(current_token_ids.device) 
        return F.log_softmax(logits, dim=-1)

    def get_next_token_log_probs_doctor(self, doctor_model, current_token_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Gets the log-probabilities for the next token from a doctor model.
        Handles potential tokenizer differences between patient and doctor.

        Args:
            doctor_model: A single doctor model instance.
            current_token_ids (torch.Tensor): Token IDs from the patient's perspective.
            attention_mask (torch.Tensor, optional): Attention mask for current_token_ids.

        Returns:
            torch.Tensor: Log-probabilities for all tokens in the vocab. Shape (batch_size, vocab_size_doctor).
        """
        # This is a placeholder for actual doctor model inference
        # Important: If patient_tokenizer and doctor_tokenizer are different,
        # current_token_ids (from patient) might need re-tokenization or mapping
        # for the doctor model. This is a complex problem (sub-token alignment).
        # For this placeholder, we assume tokenizers are compatible or current_token_ids
        # are already suitable for the doctor model.
        
        print(f"Doctor model: Getting log_probs for current_token_ids shape {current_token_ids.shape}")
        
        # If doctor_model is one of our TFPO-trained DoctorModel (nn.Module)
        if hasattr(doctor_model, 'get_token_log_probs') and callable(getattr(doctor_model, 'get_token_log_probs')):
            # The DoctorModel in tfpo_tuner.py returns logits from forward(), then get_token_log_probs processes these.
            # For reward-guided decoding, we need the full distribution $\pi_r(y_{t+1}|s_t)$
            # So, we need the doctor model's forward pass returning all next token logits.
            # The current DoctorModel.forward takes `prefix_token_ids`.
            logits = doctor_model.forward(prefix_token_ids=current_token_ids, attention_mask=attention_mask)
            # logits shape: (batch_size, doctor_vocab_size)
        else: # Fallback for a generic Hugging Face model as doctor
            # outputs = doctor_model(input_ids=current_token_ids, attention_mask=attention_mask)
            # logits = outputs.logits[:, -1, :]
            # For placeholder:
            doctor_vocab_size = doctor_model.config.vocab_size if hasattr(doctor_model, 'config') and hasattr(doctor_model.config, 'vocab_size') else 30000
            batch_size = current_token_ids.shape[0]
            logits = torch.randn(batch_size, doctor_vocab_size).to(current_token_ids.device)

        return F.log_softmax(logits, dim=-1)


    def reward_guided_decode_next_token(self, current_token_ids: torch.Tensor, alpha: float = 1.0, beta_values: list[float] = None, attention_mask: torch.Tensor = None):
        """
        Performs one step of reward-guided decoding to select the next token.
        Combines patient model's base probabilities with doctor model's preference signals.
        $\pi_{	ext{decode}}(y_{t+1}\mid s_t) \;\propto\; igl[\pi_{	ext{base}}(y_{t+1}\mid s_t)igr]^{\,lpha} \;\cdot\; \prod_i igl[\pi_{r}^{(i)}(y_{t+1}\mid s_t)igr]^{\,eta_i}$

        Args:
            current_token_ids (torch.Tensor): Tensor of token IDs generated so far by the patient model. Shape (batch_size, seq_len).
            alpha (float): Weight for the patient model's probabilities.
            beta_values (list[float]): List of weights for each doctor model's preference signals. 
                                     If None, defaults to [1.0] for each doctor model.
            attention_mask (torch.Tensor, optional): Attention mask for current_token_ids.

        Returns:
            torch.Tensor: The chosen next token ID(s). Shape (batch_size,).
        """
        if beta_values is None:
            beta_values = [1.0] * len(self.doctor_models)
        
        if len(beta_values) != len(self.doctor_models):
            raise ValueError("Length of beta_values must match the number of doctor_models.")

        # Get log-probabilities from the patient model
        log_probs_patient = self.get_next_token_log_probs_patient(current_token_ids, attention_mask) # (batch_size, patient_vocab_size)
        
        combined_log_probs = alpha * log_probs_patient

        # Get log-probabilities from each doctor model and combine
        for i, doctor_model in enumerate(self.doctor_models):
            # Important: Vocabulary alignment needed here if patient and doctor vocabs differ.
            # E.g., map doctor_log_probs to patient_vocab_space or vice-versa.
            # This is a hard problem. For now, we assume they are compatible or doctor_vocab is a subset.
            # If doctor_vocab_size < patient_vocab_size, then doctor_log_probs needs padding or careful indexing.
            # If doctor_vocab_size > patient_vocab_size, then some doctor predictions are unusable.
            # Simplest assumption: Vocabs are identical.
            
            log_probs_doctor_i = self.get_next_token_log_probs_doctor(doctor_model, current_token_ids, attention_mask) # (batch_size, doctor_vocab_size_i)
            
            if log_probs_doctor_i.shape[-1] != log_probs_patient.shape[-1]:
                # This is where vocabulary mapping would be critical.
                # For this placeholder, we'll print a warning and attempt to truncate/pad (which is naive).
                print(f"Warning: Vocab size mismatch! Patient: {log_probs_patient.shape[-1]}, Doctor {i}: {log_probs_doctor_i.shape[-1]}. Alignment needed.")
                # Naive fix: if doctor vocab is smaller, pad with large negative numbers (effectively zero prob)
                if log_probs_doctor_i.shape[-1] < log_probs_patient.shape[-1]:
                    padding_size = log_probs_patient.shape[-1] - log_probs_doctor_i.shape[-1]
                    pad_tensor = torch.full((log_probs_doctor_i.shape[0], padding_size), -float('inf'), device=log_probs_doctor_i.device)
                    log_probs_doctor_i = torch.cat([log_probs_doctor_i, pad_tensor], dim=-1)
                # Naive fix: if doctor vocab is larger, truncate
                elif log_probs_doctor_i.shape[-1] > log_probs_patient.shape[-1]:
                    log_probs_doctor_i = log_probs_doctor_i[:, :log_probs_patient.shape[-1]]
            
            combined_log_probs += beta_values[i] * log_probs_doctor_i
            
        # Sample from the combined distribution
        # Using argmax for simplicity (greedy decoding). Could use sampling (e.g. torch.multinomial).
        next_token_ids = torch.argmax(combined_log_probs, dim=-1) # Shape (batch_size,)
        return next_token_ids

    def generate_sequence(self, prompt_text: str, max_length: int = 50, alpha: float = 1.0, beta_values: list[float] = None, temperature: float = 1.0, top_k: int = 0):
        """
        Generates a sequence of tokens using reward-guided decoding.

        Args:
            prompt_text (str): The initial prompt.
            max_length (int): Maximum number of tokens to generate.
            alpha (float): Weight for the patient model.
            beta_values (list[float]): Weights for the doctor model(s).
            temperature (float): Softmax temperature for sampling. Higher is more random.
            top_k (int): If > 0, filters to top_k logits before sampling.

        Returns:
            str: The generated text sequence.
        """
        self.patient_model.eval() # Set patient model to evaluation mode
        for dm in self.doctor_models:
            dm.eval() # Set doctor model(s) to evaluation mode

        # Tokenize prompt using patient_tokenizer
        # For batch generation, input_ids should be (batch_size, seq_len)
        # Here, assuming batch_size = 1 for simplicity of generation loop
        input_ids = self.patient_tokenizer.encode(prompt_text, return_tensors="pt")
        # input_ids = input_ids.to(self.device) # Ensure on correct device

        generated_token_ids = list(input_ids.squeeze().tolist()) # Store all generated token IDs

        print(f"Starting generation with prompt: '{prompt_text}'")
        print(f"Initial token IDs: {generated_token_ids}")

        with torch.no_grad(): # Disable gradient calculations during inference
            for step in range(max_length):
                current_input_ids = torch.tensor([generated_token_ids], dtype=torch.long) #.to(self.device)
                # current_input_ids shape: (1, current_seq_len)
                
                # Get log_probs from patient and doctor(s)
                log_probs_patient = self.get_next_token_log_probs_patient(current_input_ids)
                
                combined_log_probs = alpha * log_probs_patient
                
                for i, doctor_model in enumerate(self.doctor_models):
                    # As before, current_input_ids might need conversion if tokenizers differ
                    # For doctor model, we use its own tokenizer for consistency if needed,
                    # but the input `current_input_ids` are from patient's generation.
                    # This implies doctor model needs to be able to process patient model's token IDs.
                    log_probs_doctor_i = self.get_next_token_log_probs_doctor(doctor_model, current_input_ids)
                    
                    # Vocabulary alignment (naive placeholder as before)
                    if log_probs_doctor_i.shape[-1] != log_probs_patient.shape[-1]:
                        if log_probs_doctor_i.shape[-1] < log_probs_patient.shape[-1]:
                            padding_size = log_probs_patient.shape[-1] - log_probs_doctor_i.shape[-1]
                            pad_tensor = torch.full((log_probs_doctor_i.shape[0], padding_size), -float('inf'), device=log_probs_doctor_i.device)
                            log_probs_doctor_i = torch.cat([log_probs_doctor_i, pad_tensor], dim=-1)
                        elif log_probs_doctor_i.shape[-1] > log_probs_patient.shape[-1]:
                            log_probs_doctor_i = log_probs_doctor_i[:, :log_probs_patient.shape[-1]]
                            
                    combined_log_probs += (beta_values[i] if beta_values else 1.0) * log_probs_doctor_i

                # Apply temperature
                if temperature != 1.0 and temperature > 0:
                    probabilities = F.softmax(combined_log_probs / temperature, dim=-1)
                else:
                    probabilities = F.softmax(combined_log_probs, dim=-1)

                # Apply top-k filtering
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)
                    # Create a new distribution with only top-k, renormalizing
                    filtered_probs = torch.zeros_like(probabilities).scatter_(-1, top_k_indices, top_k_probs)
                    filtered_probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)
                    next_token_id = torch.multinomial(filtered_probs, num_samples=1).squeeze(1)
                else:
                    # Sample from the distribution
                    next_token_id = torch.multinomial(probabilities, num_samples=1).squeeze(1) # Shape (1,) if batch_size=1

                next_token_id_item = next_token_id.item()
                generated_token_ids.append(next_token_id_item)
                
                # Check for EOS token (use patient_tokenizer's EOS)
                if self.patient_tokenizer.eos_token_id is not None and next_token_id_item == self.patient_tokenizer.eos_token_id:
                    print(f"EOS token ({next_token_id_item}) reached at step {step+1}.")
                    break
                
                if (step + 1) % 10 == 0: # Print progress
                    print(f"Generated {step+1} tokens...")


        # Decode the full sequence of token IDs to text using patient_tokenizer
        generated_text = self.patient_tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        print(f"Finished generation. Output: '{generated_text}'")
        return generated_text


# Example Usage (can be removed or moved to an example script later)
if __name__ == '__main__':
    # This example requires placeholder models and tokenizers that mimic Hugging Face objects.
    
    class MockHFModel: # Mimics a Hugging Face model object for placeholder
        def __init__(self, vocab_size=50257, device='cpu'): # GPT2-like vocab size
            class Config:
                def __init__(self, vocab_size):
                    self.vocab_size = vocab_size
            self.config = Config(vocab_size)
            self.device = device
            print(f"MockHFModel initialized with vocab_size: {vocab_size} on device: {device}")

        def eval(self): # Mock eval mode
            print(f"MockHFModel set to eval mode.")
            pass
        
        def to(self, device): # Mock device placement
            self.device = device
            print(f"MockHFModel moved to {device}.")
            return self
        
        # Adding a forward method to MockHFModel to better simulate tfpo_tuner.DoctorModel for the example
        def forward(self, prefix_token_ids, attention_mask=None):
            print(f"MockHFModel (as Doctor): Forward pass for prefix_token_ids shape {prefix_token_ids.shape}")
            batch_size = prefix_token_ids.shape[0]
            # Return dummy logits, this model's vocab_size is self.config.vocab_size
            logits = torch.randn(batch_size, self.config.vocab_size).to(self.device)
            return logits


    class MockHFTokenizer: # Mimics a Hugging Face tokenizer
        def __init__(self, vocab_size=50257, eos_token_id=50256):
            self.vocab_size = vocab_size
            self.eos_token_id = eos_token_id
            self.bos_token_id = 50256 # Also often EOS for GPT models
            print(f"MockHFTokenizer initialized with vocab_size: {vocab_size}, EOS ID: {eos_token_id}")

        def encode(self, text, return_tensors="pt"):
            print(f"MockHFTokenizer: Encoding text '{text}'")
            # Simple tokenization: map chars to ints, limited vocab
            tokens = [min(ord(c) % self.vocab_size, self.vocab_size-1) for c in text]
            if not tokens: tokens = [self.bos_token_id if self.bos_token_id is not None else 0] # Handle empty string
            if return_tensors == "pt":
                return torch.tensor([tokens], dtype=torch.long)
            return tokens

        def decode(self, token_ids, skip_special_tokens=True):
            print(f"MockHFTokenizer: Decoding token IDs {token_ids}")
            # Simple decoding: map ints back to chars
            chars = []
            for tid in token_ids:
                if skip_special_tokens and tid == self.eos_token_id:
                    continue
                chars.append(chr(97 + (tid % 26))) # Map to a-z for simplicity
            return "".join(chars)

    # --- Setup for example ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Patient model (larger vocab)
    patient_vocab_size = 1000
    patient_model_mock = MockHFModel(vocab_size=patient_vocab_size).to(device)
    patient_tokenizer_mock = MockHFTokenizer(vocab_size=patient_vocab_size, eos_token_id=patient_vocab_size-1)
    
    doctor_vocab_size = 1000 
    # For the example, we want get_next_token_log_probs_doctor to use the `if hasattr(doctor_model, 'get_token_log_probs')` path
    # if we were using a real tfpo_tuner.DoctorModel.
    # However, the current tfpo_tuner.DoctorModel doesn't have `config.vocab_size` directly,
    # and its `get_token_log_probs` is for specific next tokens, not the full distribution needed here.
    # The `OnlineAlignment.get_next_token_log_probs_doctor` was modified to call `doctor_model.forward()`
    # which is present in our `tfpo_tuner.DoctorModel`.
    # Let's use a modified MockHFModel that simulates this structure for the doctor model.
    # It already has `forward` and `config.vocab_size`.
    doctor_model_mock1 = MockHFModel(vocab_size=doctor_vocab_size, device=device) # This mock now has .forward()
    # To make it even closer to tfpo_tuner.DoctorModel for the purpose of get_next_token_log_probs_doctor's conditional check:
    # we can add a dummy get_token_log_probs attribute to doctor_model_mock1.
    # This will ensure the `if hasattr(doctor_model, 'get_token_log_probs')` branch is taken.
    def dummy_get_log_probs(prefix_token_ids, next_token_id, attention_mask=None): # Matches signature
        pass # Not actually used by the logic in get_next_token_log_probs_doctor, which calls .forward()
    doctor_model_mock1.get_token_log_probs = dummy_get_log_probs
    
    doctor_tokenizer_mock = MockHFTokenizer(vocab_size=doctor_vocab_size, eos_token_id=doctor_vocab_size-1)

    # Initialize OnlineAlignment
    online_aligner = OnlineAlignment(
        patient_model=patient_model_mock, 
        doctor_models=[doctor_model_mock1],
        patient_tokenizer=patient_tokenizer_mock,
        doctor_tokenizer=doctor_tokenizer_mock
    )

    # Example: Generate a sequence
    prompt = "Once upon a time"
    print(f"\n--- Test 1: Generating sequence for prompt: '{prompt}' ---")
    generated_sequence = online_aligner.generate_sequence(
        prompt_text=prompt, 
        max_length=20, 
        alpha=1.0, 
        beta_values=[0.5], 
        temperature=0.7,
        top_k=50
    )
    print(f"Final Generated Text (Test 1): {generated_sequence}")

    print(f"\n--- Test 2: Generating sequence with different alpha/beta ---")
    generated_sequence_2 = online_aligner.generate_sequence(
        prompt_text="The meaning of life is", 
        max_length=15, 
        alpha=0.7, 
        beta_values=[1.0],
        temperature=1.0,
        top_k=0 
    )
    print(f"Final Generated Text (Test 2): {generated_sequence_2}")

    print(f"\n--- Test 3: Direct call to reward_guided_decode_next_token ---")
    dummy_input_ids = patient_tokenizer_mock.encode("Test", return_tensors="pt").to(device)
    next_token = online_aligner.reward_guided_decode_next_token(
        dummy_input_ids, 
        alpha=1.0, 
        beta_values=[0.5]
    )
    print(f"Next token ID from reward_guided_decode_next_token: {next_token.item()}")
    print(f"Decoded next token: {patient_tokenizer_mock.decode([next_token.item()])}")
    
    print("\nNote: This example uses mock models and tokenizers. Vocabulary alignment is handled naively.")
    print("Actual model inference and tokenizer behavior will be more complex.")
    print("The MockHFModel for the doctor was adapted to have a `.forward()` and a dummy `.get_token_log_probs()` " + \
          "to test the preferred path in `get_next_token_log_probs_doctor`.")
