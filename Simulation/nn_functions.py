import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.distributions.uniform as urand

from torch.amp import autocast, GradScaler
from contextlib import nullcontext


def convert_to_real_loss(loss: list, norm_scales: list) -> np.array:
    '''
    Function to convert the trainiment loss to the real loss, considering the normalization of the data in FrequencyCombDataset

    This function is based on the normalization Min-Max formula: x_norm = (x - min)/(max - min)
    norm_loss = (x1_norm - x2_norm)^2 = (x1 - x2)^2/(max - min)^2 = real_loss/(max - min)^2

    Parameters:
    loss: list
        Normalized loss
    norm_scales: list
        Normalization scales of the data containing the min and max values: [min, max] 

    Returns:
    real_loss: numpy array
        Denormalized loss
    '''

    loss = np.array(loss)
    denorm_loss = loss * (norm_scales[1] - norm_scales[0])**2 

    return denorm_loss.squeeze()


def plot_training_progress(train_losses: list, val_losses: list, title: str = "Training and Validation Losses", ylabel: str = "Loss", average_curves: bool = False, M: int = 200):

    '''
    Function to plot the training and validation losses

    Parameters:
    train_losses: list
        List of training losses
    val_losses: list
        List of validation losses
    title: str
        Title of the plot
    ylabel: str
        Label of the y-axis
    average_curves: bool
        If True, plot the moving average of the curves
    M: int
        Window size of the moving average

    '''

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
    #plt.grid(which='both', alpha=0.5)
    #plt.minorticks_on()
    plt.show()


def plot_training_progress_style(train_losses: list, val_losses: list, title: str = fr"$MSE\; de\; Treinamento\; e\; Validação$", ylabel: str = r"$Erro\; (dB/Hz)^2$",ylim: tuple = (0,10), average_curves: bool = False, M: int = 200, figname: str = "training_progress.png"):
    '''
    Function to plot the training and validation losses with the IEEE style

    Parameters:
    train_losses: list
        List of training losses
    val_losses: list
        List of validation losses
    title: str
        Title of the plot
    ylabel: str
        Label of the y-axis
    average_curves: bool
        If True, plot the moving average of the curves
    M: int
        Window size of the moving average
    figname: str
        Name of the figure file
    
    '''

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
        ax.set_xlabel(r'$Épocas\times 100$', fontsize = 10)
        ax.set_ylabel(ylabel, fontsize = 10)
        ax.legend(fontsize = 8)
        ax.tick_params(labelsize=9)
        ax.set_ylim(ylim)
        ax.set_xlim(0, len(train_losses)/100)

        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
        fig.savefig(figname, dpi=300)
        plt.show()
        plt.close()


def get_num_params(model: nn.Module) -> int:
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

def calc_num_params(architecture: list) -> int:
    '''
    Function to calculate the number of parameters of a neural network model from the architecture list

    Formula: for architecture of [I, H1, H2, O], the number of parameters of the model is = (I + 1)*H1 + (H1 + 1)*H2 + (H2 + 1)*O

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

def get_architecture(loaded_dict_data: dict) -> list:
    '''
    Get the architecture of the model from the loaded dictionary data
    
    Parameters:
    loaded_dict_data: dict
    dictionary with the model data

    Returns:
    architecture: list
    list with the architecture of the model
    '''
    architecture = []
    model_state = loaded_dict_data["model_state_dict"]
    for key in model_state.keys():
        if "weight" in key:
            architecture.append(model_state[key].shape[1])
    architecture.append(model_state[key].shape[0])
    return architecture



def plot_comparison_style(target, output, freqs_GHz, loss, figname, title, ylim = (-35,35), xlabel = r"$Frequência\; Básica\; (em\; unidades\; de\; f_m)$", ylabel = r"$PSD\; (dB/Hz)$", show_max_min = False):
    
    '''
    Function to plot the comparison between the target and the output of the model with the IEEE style

    Parameters:
    target: array_like
        Target data
    output: array_like
        Output data
    freqs_GHz: array_like
        Frequency array in GHz
    loss: float
        Loss value
    figname: str
        Name of the figure file
    title: str
        Title of the plot
    ylim: tuple
        Y-axis limits
    xlabel: str
        Label of the x-axis
    ylabel: str
        Label of the y-axis
    show_max_min: bool
        If True, show the difference between the maximum and minimum values of the output

    '''
    
    with plt.style.context(['science', 'ieee', "grid", 'no-latex']):
        fig, ax = plt.subplots()
        ax.plot(freqs_GHz, target, "s", label=r'$Alvo$', markersize=4)
        ax.plot(freqs_GHz, output, "o", label=r'$Predição$', markersize=4)
        ax.legend(fontsize = 8)
        ax.autoscale(tight=True)
        ax.set_xlabel(xlabel, fontsize = 8)
        ax.set_ylabel(ylabel, fontsize = 8)
        #ax.set_xticks(freqs_GHz)
        ax.set_xlim(freqs_GHz[0]-1,freqs_GHz[-1]+1)
        ax.set_title(title)
        ax.set_xlim(freqs_GHz[0]-1,freqs_GHz[-1]+1)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=8) 

        text = fr"MSE: {loss:.3f} $(dB/Hz)^2$"
        if show_max_min:
            text += "\n" + fr"Max - Min: {np.max(output) - np.min(output):.3f} $dB$"
        ax.text(0, ylim[0]*0.97, text, ha = 'center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='silver', boxstyle='round,pad=0.3'), fontsize=8)
        fig.savefig(figname, dpi=300)
        plt.show()
        plt.close()



def run_one_epoch_forward(mode: str, loader, model, loss_fn: torch.nn.modules.loss._Loss, device: str = "cpu", optimizer: torch.optim.Optimizer = None, use_gradient_clipping: bool = False, use_autocast: bool = False, scaler: GradScaler = None) -> tuple:
    '''
    Function to run one epoch of the forward model

    Parameters:
    mode: str
        Mode of the model: 'train', 'val' or 'test'
    loader: torch DataLoader
        DataLoader of the dataset
    model: torch model
        Neural network model
    loss_fn: torch loss function
        Loss function
    device: str
        Device to run the model
    optimizer: torch optimizer
        Optimizer of the model
    gradient_clipping: bool
        If True, apply gradient clipping
    use_autocast: bool
        If True, use autocast for mixed precision training
    scaler: torch.amp.GradScaler
        GradScaler for mixed precision training

    Returns:
    avg_loss: float
        Average loss of the epoch
    outputs: array_like
        Outputs of the model
    targets: array_like
        Targets of the model
    '''

    if mode == 'train':
        model.train()
    elif mode in ['val', 'test']:
        model.eval()
    else:
        raise ValueError("Invalid mode. Try 'train', 'val' or 'test'.")

    total_loss = 0.0
    n_loops = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        if use_autocast:
            context = autocast()
        else:
            context = nullcontext()

        with context:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        if mode == 'train':
            optimizer.zero_grad()
            if scaler is not None and use_autocast:
                scaler.scale(loss).backward()
                if use_gradient_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        n_loops += 1

    avg_loss = total_loss / n_loops
    return avg_loss, outputs, targets

def run_one_epoch_inverse(mode: str, loader, forward_model, inverse_model, loss_fn: torch.nn.modules.loss, device: str = "cpu", optimizer: torch.optim = None, use_gradient_clipping: bool = False, use_autocast: bool = False, scaler: GradScaler = None) -> tuple:

    '''
    Function to run one epoch of the inverse model

    Parameters:
    mode: str
        Mode of the model: 'train', 'val' or 'test'
    loader: torch DataLoader
        DataLoader of the dataset
    forward_model: torch model
        Frozen forward model to be used in the inverse model training
    inverse_model: torch model
        Inverse model
    loss_fn: torch loss function
        Loss function
    device: str
        Device to run the model 
    optimizer: torch optimizer
        Optimizer of the model
    use_gradient_clipping: bool
        If True, apply gradient clipping
    use_autocast: bool
        If True, use autocast for mixed precision training 
    scaler: torch.amp.GradScaler
        GradScaler for mixed precision training

    Returns:
    avg_loss: float
        Average loss of the epoch
    forward_outputs: array_like
        Outputs of the forward model
    inverse_outputs: array_like
        Outputs of the inverse model
    targets: array_like
        Targets of the model
    inputs: array_like
        Inputs of the model
    '''

    if mode == 'train':
        inverse_model.train()
    elif mode == 'val' or mode == "test":
        inverse_model.eval()
    else:
        raise ValueError("Invalide mode. Try to use 'train', 'val' or 'test'.")

    total_loss = 0.0
    n_loops = 0
    for inputs, targets in loader:

        if use_autocast:
            context = autocast()
        else:
            context = nullcontext()

        with context:
            inverse_outputs = inverse_model(targets) # Forward pass through the inverse model
            forward_outputs = forward_model(inverse_outputs) # Forward pass through the forward model
            loss = loss_fn(forward_outputs, targets) # Calculate loss

        if mode == 'train':
            optimizer.zero_grad() # Reset gradients tensors
            if scaler is not None and use_autocast:
                scaler.scale(loss).backward()
                if use_gradient_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(inverse_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward() # Calculate gradients
                if use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(inverse_model.parameters(), max_norm=1.0)
                optimizer.step() # Update weights

        total_loss += loss.item()
        n_loops += 1

    avg_loss = total_loss / n_loops
    return avg_loss, forward_outputs, inverse_outputs, targets, inputs


def run_one_epoch_inverse_DiffGen(mode: str, loader, diff_ofc_gen, ofc_args, inverse_model, loss_fn: torch.nn.modules.loss, device: str = "cpu", optimizer: torch.optim = None, use_gradient_clipping: bool = False, use_autocast: bool = False, scaler: GradScaler = None) -> tuple:

    '''
    Function to run one epoch of the inverse model with a Differentiable OFC Generator instead of the forward model

    Parameters:
    mode: str
        Mode of the model: 'train', 'val' or 'test'
    loader: torch DataLoader
        DataLoader of the dataset
    diff_ofc_gen: function
        Function to calculate the forward model outputs using torch tensors
    ofc_args: list
        Arguments of the forward function
    inverse_model: torch model
        Inverse model to be trained
    loss_fn: torch loss function
        Loss function
    device: str
        Device to run the model
    optimizer: torch optimizer  
        Optimizer of the model
    use_gradient_clipping: bool
        If True, apply gradient clipping
    use_autocast: bool
        If True, use autocast for mixed precision training
    scaler: torch.amp.GradScaler
        GradScaler for mixed precision training

    Returns:
    avg_loss: float
        Average loss of the epoch
    forward_outputs: array_like
        Outputs of the forward model
    inverse_outputs: array_like
        Outputs of the inverse model
    targets: array_like 
        Targets of the model
    inputs: array_like
        Inputs of the model

    '''
    if mode == 'train':
        inverse_model.train()
    elif mode == 'val' or mode == "test":
        inverse_model.eval()
    else:
        raise ValueError("Invalid mode. Try to use 'train', 'val' or 'test'.")

    total_loss = 0.0
    n_loops = 0
    
    for inputs, targets in loader:

        if use_autocast:
            context = autocast()
        else:
            context = nullcontext()

        with context:

            inverse_outputs = inverse_model(targets)  # Forward pass through the inverse model
            forward_outputs = diff_ofc_gen(inverse_outputs.to('cpu'), ofc_args.t, ofc_args.Rs, ofc_args.Vpi, ofc_args.NFFT, ofc_args.Fa, ofc_args.SpS, ofc_args.n_peaks).to(device)  # analitical function
            if loader.dataset.zero_mean:
                forward_outputs = forward_outputs - torch.mean(forward_outputs, dim=-1, keepdim=True)
            forward_outputs = loader.dataset.normalize(forward_outputs)

            loss = loss_fn(forward_outputs, targets) # Calculate loss


        if mode == 'train':
            optimizer.zero_grad() # Reset gradients tensors
            if scaler is not None and use_autocast:
                scaler.scale(loss).backward()
                if use_gradient_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(inverse_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward() # Calculate gradients
                if use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(inverse_model.parameters(), max_norm=1.0)
                optimizer.step() # Update weights
        
        total_loss += loss.item()
        n_loops += 1

    avg_loss = total_loss / n_loops

    # Return outputs for analysis if needed
    return avg_loss, forward_outputs, inverse_outputs, targets, inputs

"""
class FrequencyCombNet(nn.Module):

    '''
    Neural network model class

    Parameters:
    architecture: list
        Architecture of the neural network model

    Methods:
    forward(self, x)
        Forward pass of the neural network model
    '''
    
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
#"""

#'''
class FrequencyCombNet(nn.Module):
    """
    Neural network model with flexible architecture and optimization-friendly features.
    
    Parameters:
    -----------
    architecture : list
        Network architecture (e.g., [input_dim, hidden_dim1, ..., output_dim])
    activation : str (default: 'relu')
        Activation function ('relu', 'leaky_relu', 'selu', 'tanh')
    use_batchnorm : bool (default: False)
        Whether to use batch normalization
    dropout_rate : float (default: 0.0)
        Dropout rate (0.0 means no dropout)
    """
    
    def __init__(self, architecture, activation='relu', use_batchnorm=False, dropout_rate=0.0, init_weights=False):
        super(FrequencyCombNet, self).__init__()
        self.architecture = architecture
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        
        layers = []
        
        # Input to first hidden layer
        layers.append(nn.Linear(architecture[0], architecture[1]))
        if self.use_batchnorm:
            layers.append(nn.BatchNorm1d(architecture[1]))
        
        # Hidden layers
        for i in range(1, len(architecture) - 1):
            layers.append(self._get_activation())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(architecture[i], architecture[i+1]))
            if self.use_batchnorm and i < len(architecture) - 2:  # No batchnorm before output
                layers.append(nn.BatchNorm1d(architecture[i+1]))
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights
        if init_weights:
            self._initialize_weights()

    def _get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(0.01)
        elif self.activation == 'selu':
            return nn.SELU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=self.activation)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)
#'''

class FrequencyCombDataset(Dataset):

    '''
    Custom dataset class to generate the data for the neural network training

    Parameters:
    function: function
        Function to generate the outputs of the dataset
    nsamples: int
        Number of samples of the dataset
    ofc_args: list
        Arguments of the function
    bounds: list
        Bounds of the inputs
    device: str
        Device to run the model
    norm_scales: list
        Normalization scales of the data containing the min and max values: [min, max]
    zero_mean: bool
        If True, zero mean the data
    creation_batch_size: int
        Batch size to generate the outputs

    Methods:
    __len__(self)
        Return the number of samples of the dataset
    make_inputs(self)
        Generate the input tensors of the dataset
    make_outputs(self, creation_batch_size)
        Generate the output tensors of the dataset
    data_size(self)
        Calculate the size of the data in bytes
    normalize(self, tensor)
        Normalize the data
    denormalize(self, tensor)
        Denormalize the data
    __getitem__(self, idx)
        Get the input and output of the dataset in the index idx
    '''
    def __init__(self, *args, device = "cpu", norm_scales = None, norm = True, zero_mean = True, creation_batch_size = 1000):
        
        self.device = device
        self.norm_scales = norm_scales
        self.norm = norm
        self.zero_mean = zero_mean
        
        # First way to create the dataset: using the function, nsamples, ofc_args and bounds
        if len(args) == 4:
            function, nsamples, ofc_args, bounds = args
            
            self.function = function
            self.nsamples = nsamples
            self.ofc_args = ofc_args
            self.bounds = bounds

            # Generate inputs
            self.input_tensors = self.make_inputs()

            # Generate outputs using batch processing
            self.output_tensors = self.make_outputs(creation_batch_size).to(self.device)

            self.input_tensors = self.input_tensors.to(self.device)

        # Second way to create the dataset: using the inputs and outputs already generated
        elif len(args) == 2:
            inputs, outputs = args

            if not torch.is_tensor(inputs):
                inputs = torch.as_tensor(inputs, dtype=torch.float32)
            if not torch.is_tensor(outputs):
                outputs = torch.as_tensor(outputs, dtype=torch.float32)

            self.input_tensors = inputs.to(self.device)
            self.output_tensors = outputs.to(self.device)

        else:
            raise ValueError("Invalid number of arguments. Try to use 2 (inputs and outputs) or 4 (function, nsamples, ofc_args, bounds).")
        
        # Zero mean the data
        if zero_mean:
            self.output_tensors -= torch.mean(self.output_tensors, dim=1).unsqueeze(1)

        # Calculate the normalization scales
        if norm_scales == None:
            self.calc_norm_scales()
        
        # Normalize the data
        if norm:
            self.output_tensors = self.normalize(self.output_tensors)

    def calc_norm_scales(self):
        min = torch.ceil(torch.min(self.output_tensors)).item()
        max = torch.ceil(torch.max(self.output_tensors)).item()
        self.norm_scales = [min, max]

    def split_train_val_test(self, train_ratio = 0.8, val_ratio = 0.1, test_ratio = 0.1, shuffle = True):
        '''
        This function returns three FrequencyCombDataset objects: train, validation and test datasets
        '''
        if train_ratio + val_ratio + test_ratio > 1:
            raise ValueError("The sum of the ratios must be less than or equal to 1.")

        len_data = self.__len__()
        train_idx = int((train_ratio)*len_data)
        val_idx   = int((train_ratio + val_ratio)*len_data)
        test_idx  = int((train_ratio + val_ratio + test_ratio)*len_data)

        train_idxs = torch.arange(0, train_idx)
        val_idxs = torch.arange(train_idx, val_idx)
        test_idxs = torch.arange(val_idx, test_idx)

        idx_shuffle = torch.arange(len_data)
        if shuffle:
            idx_shuffle = torch.randperm(len_data)

        # train, validation and test data
        train_in, train_out = self.input_tensors[idx_shuffle][train_idxs], self.output_tensors[idx_shuffle][train_idxs]
        val_in, val_out     = self.input_tensors[idx_shuffle][val_idxs],   self.output_tensors[idx_shuffle][val_idxs]
        test_in, test_out   = self.input_tensors[idx_shuffle][test_idxs],  self.output_tensors[idx_shuffle][test_idxs]

        # Create the datasets

        train_dataset = FrequencyCombDataset(train_in, train_out, device = self.device, norm_scales = self.norm_scales, norm = False, zero_mean = False)
        val_dataset   = FrequencyCombDataset(val_in,   val_out,   device = self.device, norm_scales = self.norm_scales, norm = False, zero_mean = False)
        test_dataset  = FrequencyCombDataset(test_in,  test_out,  device = self.device, norm_scales = self.norm_scales, norm = False, zero_mean = False)

        train_dataset.norm, train_dataset.zero_mean = self.norm, self.zero_mean
        val_dataset.norm, val_dataset.zero_mean = self.norm, self.zero_mean
        test_dataset.norm, test_dataset.zero_mean = self.norm, self.zero_mean

        return train_dataset, val_dataset, test_dataset
    
    def split_dataset(self, ratios, shuffle=True):
        '''
        This function returns n FrequencyCombDataset objects based on the provided ratios
        '''
        if sum(ratios) > 1:
            raise ValueError("The sum of the ratios must be less than or equal to 1.")

        len_data = self.__len__()
        indices = [int(sum(ratios[:i]) * len_data) for i in range(1, len(ratios) + 1)]

        idx_ranges = [torch.arange(0, indices[0])] + [torch.arange(indices[i-1], indices[i]) for i in range(1, len(indices))]

        idx_shuffle = torch.arange(len_data)
        if shuffle:
            idx_shuffle = torch.randperm(len_data)

        datasets = []
        for idx_range in idx_ranges:
            input_tensors, output_tensors = self.input_tensors[idx_shuffle][idx_range], self.output_tensors[idx_shuffle][idx_range]
            dataset = FrequencyCombDataset(input_tensors, output_tensors, device=self.device, norm_scales=self.norm_scales, norm=False, zero_mean=False)
            dataset.norm, dataset.zero_mean = self.norm, self.zero_mean
            datasets.append(dataset)

        return datasets
    
    def concat_with(self, *datasets, recalc_norm_scales = True):
        '''
        This function concatenates the datasets in the FrequencyCombDataset object
        '''

        self.input_tensors = torch.cat([self.input_tensors] + [dataset.input_tensors for dataset in datasets], dim=0)
        self.output_tensors = torch.cat([self.denormalize(self.output_tensors)] + [dataset.denormalize(dataset.output_tensors) for dataset in datasets], dim=0)

        #self.input_tensors = torch.cat([self.input_tensors, new_input_tensors], dim=0)
        #self.output_tensors = torch.cat([self.denormalize(self.output_tensors), new_output_tensors], dim=0)

        # Zero mean the data
        if self.zero_mean:
            self.output_tensors -= torch.mean(self.output_tensors, dim=1).unsqueeze(1)

        # Recalculate the normalization scales
        if recalc_norm_scales:
            self.calc_norm_scales()

        # Normalize the data
        if self.norm:
            self.output_tensors = self.normalize(self.output_tensors)

    def __len__(self):
        return len(self.input_tensors)
    
    def make_inputs(self):
        input_tensors = [[urand.Uniform(low, high).sample().item() for low, high in self.bounds] for _ in range(self.nsamples)]
        input_tensors = torch.as_tensor(input_tensors, dtype=torch.float32)
        return input_tensors
    
    def make_outputs(self, creation_batch_size = 1000):
        output_tensors_list = []
    
        for i in range(0, self.input_tensors.size(0), creation_batch_size):
            batch = self.input_tensors[i:i + creation_batch_size]
            output_tensors = self.function(batch, self.ofc_args.t, self.ofc_args.Rs, self.ofc_args.Vpi, self.ofc_args.NFFT, self.ofc_args.Fa, self.ofc_args.SpS, self.ofc_args.n_peaks)
            output_tensors_list.append(output_tensors)
        
        output_tensors = torch.cat(output_tensors_list, dim=0)
        return output_tensors
    
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
