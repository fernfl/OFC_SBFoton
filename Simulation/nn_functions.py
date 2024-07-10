import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.distributions.uniform as urand

def split_train_val_test_data(inputs, outputs, train_ratio = 0.8, val_ratio = 0.1, test_ratio = 0.1, shuffle = False):

    '''
    Function to split the data into training, validation and test datasets

    Parameters:
    inputs: array_like
        Inputs of the data
    outputs: array_like
        Outputs of the data
    train_ratio: float
        Ratio of the training data
    val_ratio: float
        Ratio of the validation data
    test_ratio: float
        Ratio of the test data
    shuffle: bool
        Shuffle the data before splitting
    '''

    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("The sum of the ratios must be equal to 1")

    train_idx = int((train_ratio)*len(inputs))
    val_idx   = int((train_ratio + val_ratio)*len(inputs))
    test_idx  = int((train_ratio + val_ratio + test_ratio)*len(inputs))

    train_idxs = np.arange(0, train_idx)
    val_idxs = np.arange(train_idx, val_idx)
    test_idxs = np.arange(val_idx, test_idx)

    if shuffle:
        idx = np.random.permutation(len(inputs))
        inputs = inputs[idx]
        outputs = outputs[idx]

    # train, validation and test data
    train_in, train_out = inputs[train_idxs], outputs[train_idxs]
    val_in, val_out  = inputs[val_idxs], outputs[val_idxs]
    test_in, test_out = inputs[test_idxs], outputs[test_idxs]

    return train_in, train_out, val_in, val_out, test_in, test_out


def convert_to_real_loss(loss, norm_scales):
    '''
    Function to convert the trainiment loss to the real loss, considering the normalization of the data in FrequencyCombDataset

    This function is based on the normalization Min-Max formula: x_norm = (x - min)/(max - min)
    norm_loss = (x1_norm - x2_norm)^2 = (x1 - x2)^2/(max - min)^2 = real_loss/(max - min)^2

    Parameters:
    loss: array_like
        Training loss
    norm_scale: float
        Normalization scale of the data

    '''
    loss = np.array(loss)
    loss = loss * (norm_scales[1] - norm_scales[0])**2 #
    return loss.squeeze()


def plot_training_progress(train_losses, val_losses, title = "Training and Validation Losses", ylabel = "Loss", average_curves = False, M = 200):

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    plt.figure(figsize=(15,5))
    plt.plot(train_losses, label=f'Training Loss: {train_losses[-1]:.4f}', color = "C0")
    plt.plot(val_losses, label=f'Validation Loss: {val_losses[-1]:.4f}', color='C3')

    if train_losses.size > M and average_curves:
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'valid')/w
        plt.plot(moving_average(train_losses, M), color='blue', label='Training Loss (Moving Average)')
        plt.plot(moving_average(val_losses, M), color='red', label='Validation Loss (Moving Average)')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(which='both', alpha=0.5)
    plt.minorticks_on()
    plt.show()


def plot_training_progress_style(train_losses, val_losses, title = fr"$MSE\; de\; Treinamento\; e\; Validação$", ylabel = r"$Erro\; (dB/Hz)^2$",ylim=(0,10), average_curves = False, M = 200, figname = "training_progress.png"):

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    with plt.style.context(['science', 'ieee', "grid", 'no-latex']):
        fig, ax = plt.subplots(figsize=(5*0.8,2.4*0.8), dpi = 300)
        ax.plot(np.arange(0,train_losses.size,1)/100, val_losses, label=fr'$MSE\; de\; Validação:\; ${val_losses[-1]:.4f} $(dB/Hz)^2$', color='C1')
        ax.plot(np.arange(0,train_losses.size,1)/100, train_losses, label=fr'$MSE\; de\; Treinamento:\; ${train_losses[-1]:.4f} $(dB/Hz)^2$', color = "C2")
        #ax.plot(np.arange(0,train_losses.size,1)/100, val_losses, label=fr'Validation Loss: {2.1854:.4f} $(dB/Hz)^2$', color='C1')
        #ax.plot(np.arange(0,train_losses.size,1)/100, train_losses, label=fr'Training Loss: {1.9625:.4f} $(dB/Hz)^2$', color = "C2")
        if train_losses.size > M and average_curves:
            def moving_average(x, w):
                return np.convolve(x, np.ones(w), 'valid')/w
            ax.plot(np.arange(0,train_losses.size)/100,moving_average(train_losses, M), color='blue', label='$MSE\; de\; Validação\; (Média\; Móvel)$')
            ax.plot(np.arange(0,train_losses.size)/100,moving_average(val_losses, M), color='red', label='$MSE\; de\; Treinamento\; (Média\; Móvel)$')

        ax.autoscale(tight=True)

        ax.set_title(title)
        ax.set_xlabel(r'$Épocas\times 100$')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_ylim(ylim)
        ax.set_xlim(0, len(train_losses)/100)

        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
        fig.savefig(figname, dpi=300)
        plt.show()
        plt.close()


def get_num_params(model):
    '''
    Function to calculate the number of parameters of a neural network model from the torch model

    Parameters:
    model: torch model
        Neural network model

    Returns:
    n_params: int
        Number of parameters of the neural network model
    '''
    n_params = 0
    for param in model.parameters():
        n_params += param.numel()
    return n_params

def calc_num_params(architecture):
    '''
    Function to calculate the number of parameters of a neural network model from the architecture list

    Formula: for architecture of [I, H1, H2, O], the number of parameters of the model is = I*(H1+1) + H1*(H2+1) + H2*(O+1)

    Parameters:
    architecture: list
        Architecture of the neural network model

    Returns:
    n_params: int
        Number of parameters of the neural network model
    '''
    n_params = 0
    for i in range(len(architecture) - 1):
        n_params += architecture[i]*architecture[i + 1] + architecture[i + 1]
    return n_params

def get_architecture(loaded_dict_data):
    '''
    Get the architecture of the model from the loaded dictionary data
    
    Parameters:
    loaded_dict_data: dictionary with the model data

    Returns:
    architecture: list with the architecture of the model
    '''
    architecture = []
    model_state = loaded_dict_data["model_state_dict"]
    for key in model_state.keys():
        if "weight" in key:
            architecture.append(model_state[key].shape[1])
    architecture.append(model_state[key].shape[0])
    return architecture


def plot_comparison(target, output, freqs_GHz, loss, figname, title, ylim = (-35,35), xlabel = "Frequency (GHz) - Normalized to symbol rate", ylabel = "Power Spectral Density (dB/Hz)"):
    with plt.style.context(['science', 'ieee', "grid", 'no-latex']):
        fig, ax = plt.subplots()
        ax.plot(freqs_GHz, target, "s", label='Target')
        ax.plot(freqs_GHz, output, "o", label='Predicted')
        ax.legend()
        ax.autoscale(tight=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(freqs_GHz)
        ax.set_title(title)
        ax.set_xlim(freqs_GHz[0]-0.5,freqs_GHz[-1]+0.5)
        ax.set_ylim(ylim)
        ax.text(0, ylim[0]*0.88, f"AVG Loss: {loss:.3f} (dB/Hz)^2\nMax - Min: {np.max(output) - np.min(output):.3f} dB", ha = 'center', bbox=dict(facecolor='white', alpha=1, edgecolor='silver', boxstyle='round,pad=0.3'))
        fig.savefig(figname, dpi=300)
        plt.show()
        plt.close()

def plot_comparison_style(target, output, freqs_GHz, loss, figname, title, ylim = (-35,35), xlabel = r"$Frequência\; Básica\; (em\; unidades\; de\; f_m)$", ylabel = r"$PSD\; (dB/Hz)$", show_max_min = False):
    with plt.style.context(['science', 'ieee', "grid", 'no-latex']):
        fig, ax = plt.subplots()
        ax.plot(freqs_GHz, target, "s", label=r'$Alvo$')
        ax.plot(freqs_GHz, output, "o", label=r'$Predição$')
        ax.legend()
        ax.autoscale(tight=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(freqs_GHz)
        ax.set_title(title)
        ax.set_xlim(freqs_GHz[0]-0.5,freqs_GHz[-1]+0.5)
        ax.set_ylim(ylim)

        text = fr"MSE: {loss:.3f} $(dB/Hz)^2$"
        if show_max_min:
            text += "\n" + fr"Max - Min: {np.max(output) - np.min(output):.3f} $dB$"
        ax.text(0, ylim[0]*0.88, text, ha = 'center', bbox=dict(facecolor='white', alpha=1, edgecolor='silver', boxstyle='round,pad=0.3'))
        fig.savefig(figname, dpi=300)
        plt.show()
        plt.close()

def run_one_epoch_forward(mode, loader, model, loss_fn, device="cpu", optimizer=None):
    if mode == 'train':
        model.train()
    elif mode == 'val' or mode == "test":
        model.eval()
    else:
        raise ValueError("Invalide mode. Try to use 'train', 'val' or 'test'.")

    total_loss = 0.0
    n_loops = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs) # Calculate outputs
        loss = loss_fn(outputs, targets) # Calculate loss
        total_loss += loss.item()

        print(inputs.device, inputs.requires_grad)
        print(targets.device, targets.requires_grad)
        print(outputs.device, outputs.requires_grad)

        print("INPUTS: ", inputs) 
        print("OUTPUTS: ", outputs)
        print("TARGETS: ", targets)
        

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        n_loops += 1

    avg_loss = total_loss / n_loops
    return avg_loss, outputs, targets

def run_one_epoch_inverse(mode, loader, forward_model, inverse_model, loss_fn, device="cpu", optimizer=None):
    if mode == 'train':
        inverse_model.train()
    elif mode == 'val' or mode == "test":
        inverse_model.eval()
    else:
        raise ValueError("Invalide mode. Try to use 'train', 'val' or 'test'.")

    total_loss = 0.0
    n_loops = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        inverse_outputs = inverse_model(targets) # Forward pass through the inverse model
        forward_outputs = forward_model(inverse_outputs) # Forward pass through the forward model

        # Calculate loss
        loss = loss_fn(forward_outputs, targets)
        total_loss += loss.item()

        if mode == 'train':
            optimizer.zero_grad()  # Reset gradients tensors
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights

        n_loops += 1

    avg_loss = total_loss / n_loops
    return avg_loss, forward_outputs, inverse_outputs, targets, inputs


def run_one_epoch_inverse2(mode, loader, forward_func, ofc_args, inverse_model, loss_fn, device="cpu", optimizer=None):
    if mode == 'train':
        inverse_model.train()
    elif mode == 'val' or mode == "test":
        inverse_model.eval()
    else:
        raise ValueError("Invalid mode. Try to use 'train', 'val' or 'test'.")
    
    total_loss = 0.0
    n_loops = 0
    forward_outputs = None
    inverse_outputs = None
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        inverse_outputs = inverse_model(targets)  # Forward pass through the inverse model

        # Forward pass through the function forward_func
        with torch.no_grad():
            intermediate_outputs = forward_func(inverse_outputs, ofc_args)
        
        forward_outputs = intermediate_outputs.clone().detach().requires_grad_(True).to(device)
        forward_outputs = forward_outputs - torch.mean(forward_outputs) * loader.dataset.zero_mean
        forward_outputs = loader.dataset.normalize(forward_outputs)

        # Calculate loss
        loss = loss_fn(forward_outputs, targets)
        total_loss += loss.item()

        if mode == 'train':
            optimizer.zero_grad()  # Reset gradients tensors
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights

        n_loops += 1

    avg_loss = total_loss / n_loops

    # Return outputs for analysis if needed
    return avg_loss, forward_outputs, inverse_outputs, targets, inputs


class FrequencyCombNet(nn.Module):
    def __init__(self, architecture):
        self.architecture = architecture
        super(FrequencyCombNet, self).__init__()
        layers = [nn.Linear(architecture[0], architecture[1])]
        for i in range(1, len(architecture) - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(architecture[i], architecture[i + 1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

# Define your custom dataset
class FrequencyCombDataset(Dataset):
    def __init__(self, function, nsamples, ofc_args, bounds, norm_scales = None, device = "cpu", zero_mean = True):
        self.function = function
        self.nsamples = nsamples
        self.ofc_args = ofc_args
        self.bounds = bounds
        self.norm_scales = norm_scales
        self.device = device
        self.zero_mean = zero_mean
        
        self.input_tensors = self.make_inputs(self.bounds, self.nsamples, self.device)
        self.output_tensors = self.make_outputs(self.input_tensors, self.function, self.device, self.zero_mean)

        if norm_scales == None:
            min = torch.ceil(torch.min(torch.abs(self.output_tensors))).item()
            max = torch.ceil(torch.max(torch.abs(self.output_tensors))).item()
            self.norm_scales = [min, max]
        self.output_tensors = self.normalize(self.output_tensors)
    
    def make_inputs(self, bounds, nsamples, device = "cpu"):
        inputs = [[urand.Uniform(low, high).sample().item() for low, high in bounds] for _ in range(nsamples)]
        inputs = torch.as_tensor(inputs, dtype = torch.float32).to(device)
        return inputs

    def make_outputs(self, inputs, function, device = "cpu", zero_mean = True):
        outputs = []
        for input in inputs:
            freq_peaks = function(input.view(1,-1), self.ofc_args)
            freq_peaks = freq_peaks - torch.mean(freq_peaks)*zero_mean
            outputs.append(freq_peaks)
        outputs = torch.stack(outputs, dim=0).to(device)
        return outputs

    def __len__(self):
        return len(self.input_tensors)
    
    def data_size(self):
        inputs_size_in_bytes = self.input_tensors.nelement() * self.input_tensors.element_size()/1024
        outputs_size_in_bytes = self.output_tensors.nelement() * self.output_tensors.element_size()/1024
        return inputs_size_in_bytes + outputs_size_in_bytes
    
    def normalize(self, tensor):
        norm_tensor = (tensor - self.norm_scales[0]) / (self.norm_scales[1] - self.norm_scales[0])
        return norm_tensor
    
    def denormalize(self, tensor):
        denorm_tensor = tensor * (self.norm_scales[1] - self.norm_scales[0]) + self.norm_scales[0]
        return denorm_tensor
    
    def __getitem__(self, idx):
        return self.input_tensors[idx], self.output_tensors[idx]