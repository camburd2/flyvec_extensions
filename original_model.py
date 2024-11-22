import torch


class OriginalModel:
    def __init__(self, N_size, kc_size, debug, device):
        self.W = torch.empty(kc_size, 2 * N_size, device=device).uniform_(0, .1)
        self.debug = debug
        self.device = device

    def predict(self, i):
        i = i.to(self.device)
        return torch.matmul(self.W, i)

    def save_checkpoint(self, path):
        torch.save(self.W, path)

    def load_checkpoint(self, path):
        self.W = torch.load(path)

    def learning(self, context_target_pair, probability_vector, learning_rate):
        # Ensure tensors are on the correct device
        context_target_pair = context_target_pair.to(self.device)
        probability_vector = probability_vector.to(self.device)
        learning_rate = torch.tensor(learning_rate, device=self.device)

        normalized_input = context_target_pair * probability_vector
        activations = torch.matmul(self.W, normalized_input)
        max_neuron = torch.argmax(activations)
        if self.debug:
            print(f"Max neuron: {max_neuron.item()}")
            print(f"Activations: {activations[max_neuron].item()}")

        max_neuron_weights = self.W[max_neuron]

        # Vectorized weight update
        update = learning_rate * (normalized_input - max_neuron_weights * normalized_input * max_neuron_weights)
        self.W[max_neuron] += update
