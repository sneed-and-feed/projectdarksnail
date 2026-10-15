import torch
from transformers import Trainer, TrainerCallback
from .surgery import SurgeryLossRamp

class TauAnnealingCallback(TrainerCallback):
    def __init__(self, initial_tau=1.0, min_tau=0.1, decay_steps=10000):
        self.initial_tau = initial_tau
        self.min_tau = min_tau
        self.decay_steps = decay_steps

    def on_step_begin(self, args, state, control, model, **kwargs):
        # Calculate tau (e.g., linear decay)
        progress = min(1.0, state.global_step / self.decay_steps)
        current_tau = self.initial_tau - progress * (self.initial_tau - self.min_tau)
        
        # Handle DDP / FSDP / DeepSpeed unwrapping safely
        unwrapped_model = model.module if hasattr(model, "module") else model
        
        # Inject the state into the config
        unwrapped_model.config.surgical_tau = current_tau

class SurgeryTrainer(Trainer):
    def __init__(self, *args, surgery_lambda_init=0.0, surgery_lambda_max=1.0, surgery_ramp_steps=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_ramp = SurgeryLossRamp(
            lambda_init=surgery_lambda_init,
            lambda_max=surgery_lambda_max,
            ramp_steps=surgery_ramp_steps
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in
        the first element.
        """
        unwrapped_model = model.module if hasattr(model, "module") else model

        # Standard forward pass
        # pass kwargs correctly to HF compute_loss if needed, but here we just pass **inputs
        # Some versions of transformers pass num_items_in_batch, etc.
        # But we must just pass inputs to model.
        outputs = model(**inputs)
        
        # We don't use .loss directly here, or if we do, we need to extract it
        if self.label_smoother is not None and "labels" in inputs:
            loss = self.label_smoother(outputs, inputs["labels"])
        else:
            # Save past state if it exists
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        from .surgery import SurgicalLlamaAttention
        
        # Process auxiliary load balancing losses directly from attention layers
        aux_losses = []
        for module in unwrapped_model.modules():
            if isinstance(module, SurgicalLlamaAttention):
                if hasattr(module, 'current_penalty') and module.current_penalty is not None:
                    aux_losses.append(module.current_penalty)
        
        if aux_losses:
            # Compute total load balancing penalty
            total_load_balance_loss = sum(aux_losses) / len(aux_losses)
            
            # The surgery_lambda_max can act as the load balancing coefficient (e.g. 0.01)
            # Remove the ramp if desired, but we can just use a fixed coefficient here.
            # Following standard MoE practices, we apply a small coefficient.
            # Assuming loss_ramp.lambda_max is the coefficient.
            coef = self.loss_ramp.lambda_max
            surgery_loss = coef * total_load_balance_loss
            
            loss = loss + surgery_loss

        return (loss, outputs) if return_outputs else loss
