import torch
import torch.nn as nn
import torch.optim as optim

# Placeholder for actual Hugging Face model imports
# from transformers import AutoModelForCausalLM, AutoTokenizer

class DoctorModel(nn.Module):
    def __init__(self, model_name_or_path="qwen1.5-3b-chat", vocab_size=151643): # Example vocab size for Qwen
        """
        Doctor Model: $\hat{\pi}_\theta(y_{t+1} | s_t)$
        This model predicts the probability of the next token given the current prefix.
        For now, this is a placeholder. It will be replaced with a Hugging Face transformer model.
        """
        super(DoctorModel, self).__init__()
        # In a real implementation, load a pre-trained model like Qwen 3B
        # self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        # For placeholder:
        self.vocab_size = vocab_size
        self.dummy_layer = nn.Linear(10, self.vocab_size) # Dummy layer for illustration
        print(f"Placeholder DoctorModel initialized. Vocab size: {self.vocab_size}")

    def forward(self, prefix_token_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            prefix_token_ids (torch.Tensor): Tensor of token IDs representing the prefix $s_t$. Shape: (batch_size, seq_len)
            attention_mask (torch.Tensor, optional): Attention mask for the prefix.

        Returns:
            torch.Tensor: Logits for the next token. Shape: (batch_size, vocab_size)
        """
        # Placeholder:
        # In a real scenario, this would be:
        # outputs = self.model(input_ids=prefix_token_ids, attention_mask=attention_mask)
        # next_token_logits = outputs.logits[:, -1, :] # Get logits for the last token in the sequence
        
        # Dummy implementation for placeholder
        print(f"DoctorModel forward pass with prefix_token_ids shape: {prefix_token_ids.shape}")
        batch_size, seq_len = prefix_token_ids.shape
        # Create a dummy hidden state based on input length or some other feature
        # This is highly simplified and not representative of a real transformer's processing.
        dummy_hidden_state = torch.randn(batch_size, 10).to(prefix_token_ids.device) # Dummy hidden state
        logits = self.dummy_layer(dummy_hidden_state) # (batch_size, vocab_size)
        return logits # Logits for the next token prediction

    def get_token_log_probs(self, prefix_token_ids: torch.Tensor, next_token_id: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Calculates $\log \hat{\pi}_\theta(y_{t+1} | s_k)$ for a specific next token.
        
        Args:
            prefix_token_ids (torch.Tensor): Tensor of token IDs representing the prefix $s_k$. Shape: (batch_size, seq_len)
            next_token_id (torch.Tensor): Tensor of the specific next token ID $y_{k+1}$. Shape: (batch_size, 1) or (batch_size,)
            attention_mask (torch.Tensor, optional): Attention mask.

        Returns:
            torch.Tensor: Log probability of the next_token_id. Shape: (batch_size,)
        """
        logits = self.forward(prefix_token_ids, attention_mask) # (batch_size, vocab_size)
        log_probs_all = torch.log_softmax(logits, dim=-1) # (batch_size, vocab_size)
        
        # Ensure next_token_id is correctly shaped for gathering
        if next_token_id.ndim == 1:
            next_token_id = next_token_id.unsqueeze(-1) # Shape: (batch_size, 1)
            
        log_prob_specific_token = torch.gather(log_probs_all, -1, next_token_id).squeeze(-1) # (batch_size,)
        return log_prob_specific_token


class ValueFunction(nn.Module):
    def __init__(self, input_size=768): # Example input size (e.g., hidden size of doctor model)
        """
        Value Function $V_\phi(s_t)$ or $V_\phi(s_t, y_{t+1})$
        Estimates the value of a token sequence prefix or a prefix plus a next token.
        This can be a separate model or a head of the doctor model.
        For now, this is a placeholder.
        """
        super(ValueFunction, self).__init__()
        # In a real implementation, this could take hidden states from the DoctorModel
        # or token embeddings as input.
        self.linear = nn.Linear(input_size, 1)
        print(f"Placeholder ValueFunction initialized with input_size: {input_size}")

    def forward(self, state_representation: torch.Tensor):
        """
        Args:
            state_representation (torch.Tensor): A representation of the state $s_t$ 
                                                 (e.g., hidden state from DoctorModel, or embedding of prefix).
                                                 Shape: (batch_size, feature_dim)

        Returns:
            torch.Tensor: Estimated value of the state. Shape: (batch_size, 1) or (batch_size,)
        """
        # Placeholder:
        print(f"ValueFunction forward pass with state_representation shape: {state_representation.shape}")
        return self.linear(state_representation).squeeze(-1) # Output shape (batch_size,)

# --- Loss Functions ---

def calculate_subtrajectory_balance_loss(
    doctor_model: DoctorModel, 
    value_function: ValueFunction,
    trajectories: list, # List of trajectories, each trajectory is a list of (prefix_ids, next_token_id, q_sm, q_sn)
    device: torch.device = torch.device("cpu")
):
    """
    Calculates the Subtrajectory Balance (SubTB) Loss ($\mathcal{L}_{	ext{SubTB}}$)
    as defined in Eq. ef{eq:tfpo_subtb_loss}.
    $\mathcal{L}_{	ext{SubTB}}(\hat{\pi}_\theta, V_\phi) = \sum_{(\tau) \in \mathcal{D}_{pref}} \sum_{0 \le m < n \le L_\tau} \left( \log \frac{Q(s_m)V_\phi(s_n)}{Q(s_n)V_\phi(s_m)} - \sum_{k=m}^{n-1} \log \hat{\pi}_\theta(y_{k+1} | s_k) \right)^2.$

    Args:
        doctor_model (DoctorModel): The doctor model $\hat{\pi}_\theta$.
        value_function (ValueFunction): The value function $V_\phi$.
        trajectories (list): A list of trajectories. Each trajectory is itself a list of tuples,
                             where each tuple represents a token step: (prefix_ids_tensor, next_token_id_tensor, token_reward_tensor).
                             $Q(s_t)$ will be derived from accumulated token_rewards.
                             `prefix_ids_tensor` are the token IDs for $s_k$.
                             `next_token_id_tensor` is the token ID for $y_{k+1}$.
        device (torch.device): Device to run calculations on.
    
    Returns:
        torch.Tensor: The SubTB loss.
    """
    total_subtb_loss = 0.0
    num_subtrajectories = 0

    for trajectory in trajectories:
        L_tau = len(trajectory) # Number of token steps in the trajectory
        if L_tau == 0:
            continue
        
        q_values = [torch.tensor(0.0, device=device)] # Q(s_0) is sum of rewards of tokens in s_0. If s_0 is empty/prompt, reward is 0.
        current_q_value = 0.0
        for k_idx in range(L_tau):
            _, _, r_yk_plus_1 = trajectory[k_idx] 
            current_q_value += r_yk_plus_1.item() 
            q_values.append(torch.tensor(current_q_value, device=device)) # q_values[t] is Q(s_t)

        v_phi_values = []
        dummy_feature_dim = 10 # Should match ValueFunction input_size if not using doctor model states
                               
        s0_representation = torch.randn(1, dummy_feature_dim).to(device) 
        v_phi_values.append(value_function(s0_representation).squeeze()) 

        for k_idx in range(L_tau):
            sk_plus_1_representation = torch.randn(1, dummy_feature_dim).to(device) 
            v_phi_values.append(value_function(sk_plus_1_representation).squeeze()) 
            
        for m in range(L_tau + 1): 
            for n in range(m + 1, L_tau + 1): 
                q_sm = q_values[m]
                q_sn = q_values[n]
                v_phi_sm = v_phi_values[m]
                v_phi_sn = v_phi_values[n]

                epsilon = 1e-9 
                if torch.abs(q_sn) < epsilon or torch.abs(v_phi_sm) < epsilon:
                    print(f"Warning: Skipping subtrajectory (m={m}, n={n}) due to near-zero Q(s_n) or V_phi(s_m). Q(s_n)={q_sn.item()}, V_phi(s_m)={v_phi_sm.item()}")
                    continue
                
                term_val = (q_sm * v_phi_sn) / (q_sn * v_phi_sm)
                if term_val <= epsilon: 
                    print(f"Warning: Skipping subtrajectory (m={m}, n={n}) due to non-positive argument for log: {term_val.item()}")
                    continue
                
                lhs = torch.log(term_val)

                sum_log_probs_pi_theta = torch.tensor(0.0, device=device)
                for k_idx in range(m, n): 
                    s_k_ids, y_k_plus_1_id, _ = trajectory[k_idx] 
                    s_k_ids = s_k_ids.to(device)
                    y_k_plus_1_id = y_k_plus_1_id.to(device)
                    
                    if s_k_ids.ndim == 1: s_k_ids = s_k_ids.unsqueeze(0) 
                    if y_k_plus_1_id.ndim == 0: y_k_plus_1_id = y_k_plus_1_id.unsqueeze(0) 
                    if y_k_plus_1_id.ndim == 1 and y_k_plus_1_id.numel() == 1 : y_k_plus_1_id = y_k_plus_1_id.unsqueeze(0)

                    log_pi_theta_k = doctor_model.get_token_log_probs(s_k_ids, y_k_plus_1_id) 
                    sum_log_probs_pi_theta += log_pi_theta_k.squeeze() 

                subtb_diff = lhs - sum_log_probs_pi_theta
                total_subtb_loss += subtb_diff.pow(2)
                num_subtrajectories += 1
    
    if num_subtrajectories == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_subtb_loss / num_subtrajectories


def calculate_value_discrimination_loss(value_function: ValueFunction, value_loss_data: list, gamma: float = 0.1, device: torch.device = torch.device("cpu")):
    """
    Calculates the Value Discrimination Loss ($\mathcal{L}_{	ext{value}}$)
    $\mathcal{L}_{	ext{value}}(V_\phi) = \max(0, \gamma - (V_\phi(s_t, y_w) - V_\phi(s_t, y_l)))$
    
    Args:
        value_function (ValueFunction): The value function $V_\phi$.
        value_loss_data (list): List of tuples. Each tuple: 
                                (state_rep_sw, state_rep_sl)
                                where state_rep_sw is representation of (s_t, y_w)
                                and state_rep_sl is representation of (s_t, y_l).
                                $y_w$ is preferred over $y_l$.
        gamma (float): Margin hyperparameter.
        device (torch.device): Device for calculations.

    Returns:
        torch.Tensor: The value discrimination loss.
    """
    total_value_loss = torch.tensor(0.0, device=device)
    count = 0
    for state_rep_sw, state_rep_sl in value_loss_data:
        state_rep_sw = state_rep_sw.to(device) 
        state_rep_sl = state_rep_sl.to(device)

        if state_rep_sw.ndim == 1: state_rep_sw = state_rep_sw.unsqueeze(0)
        if state_rep_sl.ndim == 1: state_rep_sl = state_rep_sl.unsqueeze(0)

        v_phi_sw = value_function(state_rep_sw).squeeze() 
        v_phi_sl = value_function(state_rep_sl).squeeze() 
        
        loss = torch.max(torch.tensor(0.0, device=device), gamma - (v_phi_sw - v_phi_sl))
        total_value_loss += loss
        count += 1
        
    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
        
    return total_value_loss / count

# --- Overall TFPO Training ---

class TFPOTrainer:
    def __init__(self, doctor_model: DoctorModel, value_function: ValueFunction, lambda_val: float = 0.1, learning_rate: float = 1e-4, device: torch.device = torch.device("cpu")):
        """
        Trainer for TFPO.
        """
        self.doctor_model = doctor_model.to(device)
        self.value_function = value_function.to(device)
        self.lambda_val = lambda_val
        self.device = device
        
        params = list(doctor_model.parameters()) + list(value_function.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

    def train_step(self, trajectories_data: list, value_loss_data: list, gamma_val_loss: float = 0.1):
        """
        Performs a single training step for TFPO.

        Args:
            trajectories_data (list): Data for SubTB loss. (See `calculate_subtrajectory_balance_loss`)
            value_loss_data (list): Data for value discrimination loss. (See `calculate_value_discrimination_loss`)
            gamma_val_loss (float): Margin for value discrimination loss.

        Returns:
            tuple: (total_loss, subtb_loss, val_disc_loss)
        """
        self.optimizer.zero_grad()
        
        subtb_loss = calculate_subtrajectory_balance_loss(
            self.doctor_model, 
            self.value_function, 
            trajectories_data,
            device=self.device
        )
        
        val_disc_loss = calculate_value_discrimination_loss(
            self.value_function,
            value_loss_data,
            gamma=gamma_val_loss,
            device=self.device
        )
        
        total_loss = subtb_loss + self.lambda_val * val_disc_loss
        
        if total_loss.requires_grad:
             total_loss.backward()
             self.optimizer.step()
        else:
            print("Warning: Total loss does not require grad. Skipping backward pass and optimizer step.")

        return total_loss.item(), subtb_loss.item(), val_disc_loss.item()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dummy_vocab_size = 1000 
    dummy_value_func_input_dim = 10 

    doctor_model_example = DoctorModel(vocab_size=dummy_vocab_size).to(device)
    value_function_example = ValueFunction(input_size=dummy_value_func_input_dim).to(device)
    
    s0_ids = torch.tensor([1, 2], device=device) 
    y1_id = torch.tensor([10], device=device)
    y2_id = torch.tensor([20], device=device)
    y3_id = torch.tensor([30], device=device)
    
    trajectory1 = [
        (s0_ids, y1_id, torch.tensor(0.5, device=device)),               
        (torch.cat([s0_ids, y1_id]), y2_id, torch.tensor(0.8, device=device)),          
        (torch.cat([s0_ids, y1_id, y2_id]), y3_id, torch.tensor(-0.2, device=device)) 
    ]
    
    s0_alt_ids = torch.tensor([3, 4, 5], device=device)
    y1_alt_id = torch.tensor([15], device=device)
    y2_alt_id = torch.tensor([25], device=device)
    trajectory2 = [
        (s0_alt_ids, y1_alt_id, torch.tensor(0.7, device=device)),                       
        (torch.cat([s0_alt_ids, y1_alt_id]), y2_alt_id, torch.tensor(0.1, device=device)) 
    ]
    trajectories_data_example = [trajectory1, trajectory2]

    value_loss_data_example = [
        (torch.randn(1, dummy_value_func_input_dim, device=device), torch.randn(1, dummy_value_func_input_dim, device=device)),
        (torch.randn(1, dummy_value_func_input_dim, device=device), torch.randn(1, dummy_value_func_input_dim, device=device))
    ]

    tfpo_trainer = TFPOTrainer(doctor_model_example, value_function_example, lambda_val=0.5, device=device, learning_rate=1e-3)

    print("\nStarting TFPO training step...")
    for i in range(3): # Run a few steps
        total_loss, subtb_loss, val_disc_loss = tfpo_trainer.train_step(
            trajectories_data_example, 
            value_loss_data_example,
            gamma_val_loss=0.1
        )
        print(f"Step {i+1}: Total Loss: {total_loss:.4f}, SubTB Loss: {subtb_loss:.4f}, Value Disc Loss: {val_disc_loss:.4f}")
        if subtb_loss == 0 and val_disc_loss == 0 and total_loss == 0 and not (len(trajectories_data_example) == 0 and len(value_loss_data_example) == 0) :
             print("Warning: All losses are zero. This might indicate an issue if not expected (e.g. data is not causing any loss).")
    
    print("\n--- Example Walkthrough ---")
    print("DoctorModel:")
    print("  - Takes prefix token IDs (e.g., for s_k).")
    print("  - Outputs logits for the next token (y_{k+1}).")
    print("  - `get_token_log_probs` gives log probability for a specific y_{k+1}.")
    print("ValueFunction:")
    print("  - Takes a state representation (e.g., for s_t or (s_t, y_{t+1})).")
    print("  - Outputs a scalar value V(state).")
    print("SubTB Loss:")
    print("  - Iterates trajectories (pref_data) and subtrajectories (m, n).")
    print("  - LHS: log( (Q(s_m)V(s_n)) / (Q(s_n)V(s_m)) )")
    print("     - Q(s_t) is sum of rewards of tokens making up s_t. Q(s_0)=0.")
    print("     - V(s_t) from ValueFunction (using dummy state representations for now).")
    print("  - RHS: sum log pi_theta(y_{k+1} | s_k) from DoctorModel.")
    print("Value Discrimination Loss:")
    print("  - Uses V(s_t, y_w) and V(s_t, y_l) from ValueFunction.")
    print("  - Loss = max(0, gamma - (V_win - V_lose) ).")
    print("TFPOTrainer:")
    print("  - Combines losses: SubTB_loss + lambda * ValueDisc_loss.")
    print("  - Performs backpropagation and optimizer step.")
    print("\nNote: The losses are based on placeholder models and dummy data. ")
    print("The SubTB loss calculation involves iterating through all subtrajectories (O(L^2) per trajectory).")
    print("The Q value calculation (sum of rewards in prefix) and state representations for V_phi (currently dummy) are key areas for future refinement with real models.")
    print("Ensure device consistency for all tensors and models.")
    print("The warning for zero losses will trigger if all losses are zero and there was actual data processed.")
