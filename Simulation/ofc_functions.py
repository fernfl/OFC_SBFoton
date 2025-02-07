#from optic.models.devices import mzm, pm
#from optic.utils import parameters
import numpy as np
import torch.distributions.uniform as urand
from scipy.optimize import minimize

import torch
from torch.cuda.amp import autocast

#from typing import List, Tuple

class parameters():
    pass

#@torch.jit.script
def pm(Ai: torch.Tensor, u: torch.Tensor, Vpi: float) -> torch.Tensor:
    '''
    This function calculates the output of a phase modulator (PM) given the input signal and the phase modulation signal.

    Parameters:
    Ai: torch.Tensor
        Input signal
    u: torch.Tensor
        Phase modulation signal
    Vpi: float
        Half-wave voltage of the modulator (V)

    Returns:
    Ao: torch.Tensor
        Output of the phase modulator
    '''

    K = torch.pi/Vpi
    Ao = Ai * torch.exp(1j * K * u) 
    
    return Ao

#@torch.jit.script
def mzm(Ai: torch.Tensor, u: torch.Tensor, Vpi: float, Vb: torch.Tensor) -> torch.Tensor:
    '''
    This function calculates the output of a single-drive Mach-Zehnder modulator (MZM) given the input signal and the phase modulation signal.
    Parameters:
    Ai: torch.Tensor
        Input signal
    u: torch.Tensor
        Phase modulation signal
    Vpi: float
        Half-wave voltage of the modulator (V)
    Vb: torch.Tensor
        Bias voltage of the modulator (V)
    
    Returns:
    Ao: torch.Tensor
        Output of the single-drive Mach-Zehnder modulator

    '''

    K = torch.pi/Vpi
    Ao = Ai * torch.cos(0.5 * K * (u + Vb))

    return Ao

#@torch.jit.script
def ddmzm(Ai: torch.Tensor, u1: torch.Tensor, u2: torch.Tensor, Vb1: torch.Tensor, Vb2: torch.Tensor, Vpi: float) -> torch.Tensor:
    '''
    This function calculates the output of a dual-drive Mach-Zehnder modulator (DD-MZM) given the input signal and the phase modulation signals.

    Parameters:
    Ai: torch.Tensor
        Input signal
    u1: torch.Tensor
        Phase modulation signal for the first arm
    u2: torch.Tensor
        Phase modulation signal for the second arm
    Vb1: torch.Tensor
        Bias voltage for the first arm
    Vb2: torch.Tensor 
        Bias voltage for the second arm
    Vpi: float
        Half-wave voltage of the modulator (V)

    Returns:
    Ao: torch.Tensor
        Output of the dual-drive Mach-Zehnder modulator
    '''

    K = torch.pi/Vpi
    Ao = 0.5 * Ai * (torch.exp(1j * K * (u1 + Vb1) ) + torch.exp(1j * K * (u2 + Vb2)))

    return Ao

#@torch.jit.script
def frequencyCombGenerator_PM_MZM_MZM(params: torch.Tensor, Rs: float, t: torch.Tensor, P: torch.Tensor, Vpi: float) -> torch.Tensor:

    '''
    This function generates a frequency comb signal using a PM and two MZMs.
    
    Parameters:

    params: torch.Tensor
        Parameters to generate the frequency comb signal. Contains the following values:
        V1 : float
            Amplitude of the signal for the PM modulator (V)
        V2 : float
            Amplitude of the signal for the first MZM modulator (V)
        V3 : float
            Amplitude of the signal for the second MZM modulator (V)
        Phase1 : float
            Phase of the signal for the PM modulator (rad)
        Phase2 : float
            Phase of the signal for the first MZM modulator (rad)
        Phase3 : float
            Phase of the signal for the second MZM modulator (rad)
        Vb2 : float
            Bias voltage for the first MZM modulator (V)
        Vb3 : float
            Bias voltage for the second MZM modulator (V)

    Rs: float
        Symbol rate (samples per second)
    t: torch.Tensor
        Time vector
    P: float
        Power of the frequency comb signal (W)
    Vpi: float
        Half-wave voltage of the MZM (V)

    Returns:
    frequency_comb: torch.Tensor
        Frequency comb signal
    '''

    V1, V2, V3, Phase1, Phase2, Phase3, Vb2, Vb3 = torch.split(params, 1, dim=-1)

    omega_m = 2 * torch.pi * Rs
    pi_Rs_t = omega_m * t
    u1, u2, u3 = [V * torch.cos(pi_Rs_t + Phase) for V, Phase in zip([V1, V2, V3], [Phase1, Phase2, Phase3])]

    frequency_comb = P
    frequency_comb =  pm(frequency_comb, u1, Vpi)
    frequency_comb = mzm(frequency_comb, u2, Vpi, Vb2)
    frequency_comb = mzm(frequency_comb, u3, Vpi, Vb3)

    return frequency_comb

#@torch.jit.script
def frequencyCombGenerator_PM_PM_MZM(params: torch.Tensor, Rs: float, t: torch.Tensor, P: torch.Tensor, Vpi: float) -> torch.Tensor:

    '''
    This function generates a frequency comb signal using two PMs and a MZM.
    
    Parameters:

    params: torch.Tensor
        Parameters to generate the frequency comb signal. Contains the following values:
        V1 : float
            Amplitude of the signal for the first PM modulator (V)
        V2 : float
            Amplitude of the signal for the second PM modulator (V)
        V3 : float
            Amplitude of the signal for the MZM modulator (V)
        Phase1 : float
            Phase of the signal for the first PM modulator (rad)
        Phase2 : float
            Phase of the signal for the second PM modulator (rad)
        Phase3 : float
            Phase of the signal for the MZM modulator (rad)
        Vb3 : float
            Bias voltage for the MZM modulator (V)
    Rs: float
        Symbol rate (samples per second)
    t: torch.Tensor
        Time vector
    P: float
        Power of the frequency comb signal (W)
    Vpi: float
        Half-wave voltage of the MZM (V)

    Returns:
    frequency_comb: torch.Tensor
        Frequency comb signal
    '''
    V1, V2, V3, Phase1, Phase2, Phase3, Vb3 = torch.split(params, 1, dim=-1)

    pi_Rs_t = 2 * torch.pi * Rs * t
    u1, u2, u3 = [V * torch.cos(pi_Rs_t + Phase) for V, Phase in zip([V1, V2, V3], [Phase1, Phase2, Phase3])]

    frequency_comb = P
    frequency_comb =  pm(frequency_comb, u1, Vpi)
    frequency_comb =  pm(frequency_comb, u2, Vpi)
    frequency_comb = mzm(frequency_comb, u3, Vpi, Vb3)

    return frequency_comb


#@torch.jit.script
def frequencyCombGenerator_DDMZM(params: torch.Tensor, Rs: float, t: torch.Tensor, Vpi: float) -> torch.Tensor:
    ''' 
    This function generates a frequency comb using a dual-drive Mach-Zehnder modulator (DD-MZM)

    Parameters:
    params : torch.Tensor
        Parameters to generate the frequency comb signal. Contains the following values:
        V1 : float
            Amplitude of the signal for the first modulator (V)
        V2 : float
            Amplitude of the signal for the second modulator (V)
        Phase1 : float
            Phase of the signal for the first modulator (rad)
        Phase2 : float
            Phase of the signal for the second modulator (rad)
        Vb1 : float
            Bias voltage for the first modulator (V)
        Vb2 : float
            Bias voltage for the second modulator (V)
        P : float
            Power of the frequency comb signal (W)
    Rs : float
        Symbol rate (samples per second)
    t : torch.Tensor
        Time vector
    Vpi : float
        Half-wave voltage of the DDMZM (V)

    Returns:
    frequency_comb : torch.Tensor
        Frequency comb generated by the DDMZM (W)

    '''

    V1, V2, Vb1, Vb2, P = torch.split(params, 1, dim=-1)
    Phase1, Phase2 = 0, 0

    pi_Rs_t = 2 * torch.pi * Rs * t
    u1, u2 = [V * torch.cos(pi_Rs_t + Phase) for V, Phase in zip([V1, V2], [Phase1, Phase2])]
    
    frequency_comb = ddmzm(P, u1, u2, Vb1, Vb2, Vpi)

    return frequency_comb

#@torch.jit.script
def get_psd_ByFFT(signal: torch.Tensor, Fa: float, NFFT: int = 16*1024) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Calculate the Power Spectral Density (PSD) of a signal using FFT in PyTorch.

    Parameters:
    signal: torch.Tensor
        Signal to calculate the PSD
    Fa: float
        Sampling frequency of the signal (samples per second)
    NFFT: int
        Number of points of the FFT (default is 16*1024)
        
    Returns:
    psd: torch.Tensor
        Power Spectral Density of the signal
    freqs: torch.Tensor
        Frequencies of the PSD    
    '''

    fft_result = torch.fft.fftshift(torch.fft.fft(signal, n=NFFT, dim=-1), dim=-1)  # get the fft
    power_spectrum = torch.abs(fft_result) ** 2  # get the power spectrum
    psd = power_spectrum / (NFFT * Fa)  # get the power spectral density
    psd[psd == 0] = 1.2e-7 #torch.finfo(psd.dtype).eps # to avoid log(0) issues
    freqs = torch.fft.fftshift(torch.fft.fftfreq(NFFT, 1 / Fa)).to(psd.device) # get the frequencies

    return psd, freqs


#@torch.jit.script
def get_indx_peaks(log_Pxx: torch.Tensor, SpRs: int, n_peaks: int) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Function to get the indexes of the peaks in the power spectrum of the frequency comb signal

    Parameters:
    log_Pxx: torch.Tensor
        Power spectrum of the frequency comb signal
    SpRs: int
        Samples by each Rs in the FFT domain. Each peak is separated by the Rs frequency. SpRs = (NFFT//SpS)
    n_peaks: int
        Number of peaks to be found

    Returns:
    peaks: torch.Tensor
        Peaks of the power spectrum of the frequency comb signal
    indx: torch.Tensor
        Indexes of the peaks in the power spectrum of the frequency comb signal
    '''

    center_indx = log_Pxx.size(-1) // 2  # get the center index
    offsets = torch.arange(-(n_peaks // 2), (n_peaks // 2) + 1, device=log_Pxx.device) * SpRs  # create an array of offsets for the peaks
    offsets = offsets.to(torch.int)  # Ensure offsets are integers
    indx = center_indx + offsets  # calculate the indices of the peaks
    peaks = log_Pxx[:, indx]  # extract the peaks from log_Pxx

    return peaks, indx

def frequencyCombPeaks(params: torch.Tensor, t: torch.Tensor, Rs: float, Vpi: float, NFFT: int, Fa: float, SpS: int, n_peaks: int) -> torch.Tensor:
    ''' 
    Function to get the peaks of the power spectrum of the frequency comb signal

    Parameters:
    params: torch.Tensor
        Parameters to generate the frequency comb signal. Contains the following values:
        V1 : float
            Amplitude of the signal for the first modulator (V)
        V2 : float
            Amplitude of the signal for the second modulator (V)
        V3 : float
            Amplitude of the signal for the third modulator (V)
        Phase1 : float
            Phase of the signal for the first modulator (rad)
        Phase2 : float
            Phase of the signal for the second modulator (rad)
        Phase3 : float
            Phase of the signal for the third modulator (rad)
        Vb2 : float
            Bias voltage for the second modulator (V)
        Vb3 : float
            Bias voltage for the third modulator (V)
        P : float
            Power of the frequency comb signal (W)
    t: torch.Tensor
        Time vector
    Rs: float
        Symbol rate (samples per second)
    Vpi: float
        Half-wave voltage of the MZM (V)
    P: float
        Power of the frequency comb signal (W)
    NFFT: int
        Number of points of the FFT (multiple of SpS)
    Fa: float
        Sampling frequency of the signal (samples per second)
    SpS: int
        Samples by each Rs
    n_peaks: int
        Number of peaks to be found (odd number)

    Returns:
    peaks: torch.Tensor
        Peaks of the power spectrum of the frequency comb signal
    '''
    
    frequency_comb = frequencyCombGenerator_DDMZM(params, Rs, t, Vpi) # Generate the frequency comb signal
    #frequency_comb = frequencyCombGenerator_PM_MZM_MZM(params, Rs, t, P, Vpi) # Generate the frequency comb signal
    #frequency_comb = frequencyCombGenerator_PM_PM_MZM(params, Rs, t, P, Vpi) # Generate the frequency comb signal
    Pxx, _ = get_psd_ByFFT(frequency_comb, Fa, NFFT) # Get the power spectrum of the frequency comb signal
    log_Pxx = 10*torch.log10(Pxx) # Convert the power spectrum to dB
    peaks, _ = get_indx_peaks(log_Pxx, NFFT//SpS, n_peaks) # Get the indexes of the peaks

    return peaks

#@torch.jit.script
def analytical_function_compact(params: torch.Tensor, t: torch.Tensor, Rs: float, Vpi: float, P: torch.Tensor, NFFT: int, Fa: float, SpS: int, n_peaks: int) -> torch.Tensor:

    '''
    Function to generate the frequency comb signal peaks of PM-MZM-MZM

    Parameters:
    params: torch.Tensor
        Parameters to generate the frequency comb signal. Contains the following values:
        V1 : float
            Amplitude of the signal for the PM modulator (V)
        V2 : float
            Amplitude of the signal for the second MZM modulator (V)
        V3 : float  
            Amplitude of the signal for the third MZM modulator (V)
        Phase1 : float
            Phase of the signal for the PM modulator (rad)
        Phase2 : float
            Phase of the signal for the second MZM modulator (rad)
        Phase3 : float
            Phase of the signal for the third MZM modulator (rad)
        Vb2 : float
            Bias voltage for the second MZM modulator (V)
        Vb3 : float 
            Bias voltage for the third MZM modulator (V)

    t: torch.Tensor
        Time vector
    Rs: float
        Symbol rate (samples per second)
    Vpi: float
        Half-wave voltage of the modulators (V)
    P: torch.Tensor
        Power of the frequency comb signal (W)
    NFFT: int
        Number of points of the FFT
    Fa: float
        Sampling frequency of the signal (samples per second)
    SpS: int
        Samples by each Rs
    n_peaks: int
        Number of peaks to be found

    Returns:
    peaks: torch.Tensor
        Peaks of the power spectrum of the frequency comb signal
    '''
    #V1, V2, V3, Phase1, Phase2, Phase3, Vb2, Vb3 = [p.contiguous() for p in params.T.unsqueeze(2)]
    #V1, V2, V3, Phase1, Phase2, Phase3, Vb2, Vb3 = params.T.unsqueeze(2) # Reshape each one to [N, 1] for broadcasting

    V1, V2, V3, Phase1, Phase2, Phase3, Vb2, Vb3 = torch.split(params, 1, dim=-1)

    pi_Rs_t = 2 * torch.pi * Rs * t
    K = torch.pi / Vpi

    #with autocast(enabled=True):
    u1 = V1 * torch.cos(pi_Rs_t + Phase1)
    u2 = V2 * torch.cos(pi_Rs_t + Phase2)
    u3 = V3 * torch.cos(pi_Rs_t + Phase3)
    cos_u1_K = torch.cos(u1 * K)
    sin_u1_K = torch.sin(u1 * K)
    cos_u2_Vb2_K = torch.cos((0.5 * (u2 + Vb2)) * K)
    cos_u3_Vb3_K = torch.cos((0.5 * (u3 + Vb3)) * K)
    real_part = P * cos_u1_K * cos_u2_Vb2_K * cos_u3_Vb3_K
    imag_part = P * sin_u1_K * cos_u2_Vb2_K * cos_u3_Vb3_K
    frequency_comb = torch.complex(real_part, imag_part)

    psd = torch.fft.fft(frequency_comb, n=NFFT, dim=-1)
    psd = torch.fft.fftshift(psd, dim=-1)
    psd = torch.abs(psd) ** 2 / (NFFT * Fa)
    psd[psd == 0] = 1.2e-7 #torch.finfo(psd.dtype).eps
    log_Pxx = 10 * torch.log10(psd)

    n_peaks_2 = n_peaks // 2
    indx = (log_Pxx.size(-1) // 2) + torch.arange(-n_peaks_2, n_peaks_2 + 1, device=params.device).to(torch.int) * (NFFT // SpS)
    #peaks = torch.index_select(log_Pxx, 1, indx) 
    peaks = log_Pxx[:, indx]  # extract the peaks from log_Pxx


    return peaks

#@torch.jit.script
def analytical_function_compact_DDMZM(params: torch.Tensor, t: torch.Tensor, Rs: float, Vpi: float, NFFT: int, Fa: float, SpS: int, n_peaks: int) -> torch.Tensor:

    '''
    Function to generate the frequency comb signal peaks of PM-MZM-MZM

    Parameters:
    params: torch.Tensor
        Parameters to generate the frequency comb signal. Contains the following values:
        V1 : float
            Amplitude of the signal for the first arm (V)
        V2 : float
            Amplitude of the signal for the second arm (V) 
        Vb1 : float
            Bias voltage for the first arm (V)
        Vb2 : float
            Bias voltage for the second arm (V)
        P : float
            Power of the frequency comb signal (W)

    t: torch.Tensor
        Time vector
    Rs: float
        Symbol rate (samples per second)
    Vpi: float
        Half-wave voltage of the MZM (V)
    NFFT: int
        Number of points of the FFT (multiple of SpS)
    Fa: float
        Sampling frequency of the signal (samples per second)
    SpS: int
        Samples by each Rs
    n_peaks: int
        Number of peaks to be found (odd number)

    Returns:
    peaks: torch.Tensor
        Peaks of the power spectrum of the frequency comb signal

    '''
    # calculate the frequency comb signal
    #V1, V2, Vb1, Vb2, P = [p.contiguous() for p in params.T.unsqueeze(2)]
    #V1, V2, Vb1, Vb2, P = params.T.unsqueeze(2) # Reshape each one to [N, 1] for broadcasting

    V1, V2, Vb1, Vb2, P = torch.split(params, 1, dim=-1)

    Phase1, Phase2 = 0, 0

    pi_Rs_t = 2 * torch.pi * Rs * t
    K = torch.pi / Vpi

    u1 = V1 * torch.cos(pi_Rs_t + Phase1)
    u2 = V2 * torch.cos(pi_Rs_t + Phase2)

    frequency_comb =  0.5 * P * (torch.exp(1j * K * (u1 + Vb1) ) + torch.exp(1j * K * (u2 + Vb2)))
    
    # calculate the PSD
    psd = torch.fft.fft(frequency_comb, n=NFFT, dim=-1)
    psd = torch.fft.fftshift(psd, dim=-1)
    psd = torch.abs(psd) ** 2 / (NFFT * Fa)
    psd[psd == 0] = 1.2e-7 #torch.finfo(psd.dtype).eps
    log_Pxx = 10 * torch.log10(psd)
    
    # get the peaks
    n_peaks_2 = n_peaks // 2
    indx = (log_Pxx.size(-1) // 2) + torch.arange(-n_peaks_2, n_peaks_2 + 1, device=params.device).to(torch.int) * (NFFT // SpS)
    #peaks = torch.index_select(log_Pxx, 1, indx) 
    peaks = log_Pxx[:, indx]
    
    return peaks

def analytical_function_compact_numpy(params: list, args: parameters) -> np.array:
    '''
    Function to generate the frequency comb signal peaks of PM-MZM-MZM using NumPy

    Parameters:
    params: list
        Parameters to generate the frequency comb signal. Contains the following values:
        V1 : float
            Amplitude of the signal for the PM modulator (V)
        V2 : float
            Amplitude of the signal for the first MZM modulator (V)
        V3 : float
            Amplitude of the signal for the second MZM modulator (V)
        Phase1 : float
            Phase of the signal for the PM modulator (rad)
        Phase2 : float
            Phase of the signal for the first MZM modulator (rad) 
        Phase3 : float
            Phase of the signal for the second MZM modulator (rad)
        Vb2 : float
            Bias voltage for the first MZM modulator (V)
        Vb3 : float 
            Bias voltage for the second MZM modulator (V)
    args: parameters
        Parameters object with the arguments to generate the frequency comb signal
        Should contain the following attributes:
        t: numpy array
            Time vector
        Rs: float
            Symbol rate (samples per second)
        Vpi: float
            Half-wave voltage of the MZM (V)
        P: float
            Power of the frequency comb signal (W)
        NFFT: int
            Number of points of the FFT (multiple of SpS)
        Fa: float
            Sampling frequency of the signal (samples per second)
        SpS: int
            Samples by each Rs
        n_peaks: int
            Number of peaks to be found (odd number)

    Returns:
    peaks: numpy.array
        Peaks of the power spectrum of the frequency comb signal
    '''

    V1, V2, V3, Phase1, Phase2, Phase3, Vb2, Vb3 = params

    pi_Rs_t = 2 * np.pi * args.Rs * args.t
    K = np.pi / args.Vpi
    u1 = V1 * np.cos(pi_Rs_t + Phase1)
    u2 = V2 * np.cos(pi_Rs_t + Phase2)
    u3 = V3 * np.cos(pi_Rs_t + Phase3)
    cos_u1_K, sin_u1_K = np.cos(u1 * K), np.sin(u1 * K)
    cos_u2_Vb2_K = np.cos((0.5 * (u2 + Vb2)) * K)
    cos_u3_Vb3_K = np.cos((0.5 * (u3 + Vb3)) * K)
    frequency_comb = args.P * (cos_u1_K + 1j * sin_u1_K) * cos_u2_Vb2_K * cos_u3_Vb3_K
    
    psd = np.fft.fft(frequency_comb, n=args.NFFT)
    psd = np.fft.fftshift(psd)
    psd = np.abs(psd) ** 2 / (args.NFFT * args.Fa)
    psd[psd == 0] = np.finfo(psd.dtype).eps
    log_Pxx = 10 * np.log10(psd)
    
    n_peaks_2 = args.n_peaks // 2
    indx = (log_Pxx.shape[-1] // 2) + np.arange(-n_peaks_2, n_peaks_2 + 1, dtype=int) * (args.NFFT // args.SpS)
    peaks = log_Pxx[indx]
    return peaks

def analytical_function_compact_numpy_DDMZM(params: list, args: parameters) -> np.array:
    '''
    Function to generate the frequency comb signal peaks of PM-MZM-MZM using NumPy

    Parameters:
    params: list
        Parameters to generate the frequency comb signal. Contains the following values:
        V1 : float
            Amplitude of the signal for the first arm (V)
        V2 : float
            Amplitude of the signal for the second arm (V)
        Vb1 : float
            Bias voltage for the first arm (V)
        Vb2 : float
            Bias voltage for the second arm (V)
        P : float
            Power of the frequency comb signal (W)

    args: parameters
        Parameters object with the arguments to generate the frequency comb signal
        Should contain the following attributes: 
        t: numpy array
            Time vector
        Rs: float
            Symbol rate (samples per second)
        Vpi: float
            Half-wave voltage of the MZM (V)
        NFFT: int
            Number of points of the FFT (multiple of SpS)
        Fa: float
            Sampling frequency of the signal (samples per second)
        SpS: int
            Samples by each Rs
        n_peaks: int
            Number of peaks to be found (odd number)

    Returns:
    peaks: numpy.array
        Peaks of the power spectrum of the frequency comb signal
    '''
    V1, V2, Vb1, Vb2, P = params
    Phase1, Phase2 = 0, 0
    
    pi_Rs_t = 2 * np.pi * args.Rs * args.t
    K = np.pi / args.Vpi
    
    u1 = V1 * np.cos(pi_Rs_t + Phase1)
    u2 = V2 * np.cos(pi_Rs_t + Phase2)

    frequency_comb =  0.5 * P * (np.exp(1j * K * (u1 + Vb1) ) + np.exp(1j * K * (u2 + Vb2)))
    
    # calculate the PSD
    psd = np.fft.fft(frequency_comb, n=args.NFFT)
    psd = np.fft.fftshift(psd)
    psd = np.abs(psd) ** 2 / (args.NFFT * args.Fa)
    psd[psd == 0] = np.finfo(psd.dtype).eps
    log_Pxx = 10 * np.log10(psd)
    
    # get the peaks
    n_peaks_2 = args.n_peaks // 2
    indx = (log_Pxx.shape[-1] // 2) + np.arange(-n_peaks_2, n_peaks_2 + 1, dtype=int) * (args.NFFT // args.SpS)
    peaks = log_Pxx[indx]
    return peaks

def optimization_flatComb_numpy(initial_guess: list, args: parameters, bounds: list, n_max: int = 100, method: str = "SLSQP") -> tuple[list, np.array, float]:

    ''' 
    Function to optimize the parameters of the frequency comb signal

    Parameters:
    initial_guess: list
        Initial guess for the parameters of the frequency comb signal
    args: parameters()
        Parameters object with the arguments to generate the frequency comb signal
        Should contain the following attributes: 
        t: numpy array
            Time vector
        Rs: float
            Symbol rate (samples per second)
        Vpi: float
            Half-wave voltage of the MZM (V)
        P: float
            Power of the frequency comb signal (W)
        NFFT: int
            Number of points of the FFT (multiple of SpS)
        Fa: float
            Sampling frequency of the signal (samples per second)
        SpS: int
            Samples by each Rs
        n_peaks: int
            Number of peaks to be found (odd number)

    bounds: list
        Bounds of the parameters
    n_max: int
        Max number of iterations
    method: string
        Optimization method: Nelder-Mead, Powell, L-BFGS-B, TNC, COBYLA, SLSQP, trust-constr

    Returns:
    optimized_params: list
        Optimized parameters of the frequency comb signal
    '''

    #peaks_func = analytical_function_compact_numpy
    peaks_func = analytical_function_compact_numpy_DDMZM

    def objective_function(params: list , *args: parameters) -> float:

        peaks = peaks_func(params, *args)
        var2 = np.var(peaks)**2
        
        return var2
    
    optimized_params = initial_guess
    n = 0
    while n < n_max:
        initial_guess = optimized_params
        result = minimize(objective_function, initial_guess, args = args, method=method, bounds = bounds)
        optimized_params = result.x
        
        peaks = peaks_func(optimized_params, args)
        min_max = np.max(peaks) - np.min(peaks)

        if min_max < 1:
            break
        n += 1

    return optimized_params, peaks, min_max


def create_flatCombs_numpy(nsamples: int, args: parameters, bounds: tuple, upper_min_max: float = 5.0, n_max: int = 100, method: str = "COBYLA") -> tuple[list, list, list, list]:
    '''
    Function to create the flatComb datasets using scipy.optimize.minimize and NumPy

    Parameters:
    nsamples: int
        Number of samples to be created
    args: parameters()
        Parameters object with the arguments to generate the frequency comb signal
        Should contain the following attributes: 
        t: numpy array
            Time vector
        Rs: float
            Symbol rate (samples per second)
        Vpi: float
            Half-wave voltage of the MZM (V)
        P: float
            Power of the frequency comb signal (W)
        NFFT: int
            Number of points of the FFT (multiple of SpS)
        Fa: float
            Sampling frequency of the signal (samples per second)
        SpS: int
            Samples by each Rs
        n_peaks: int
            Number of peaks to be found (odd number)
    bounds: list
        Bounds of the parameters
    upper_min_max: float
        Maximum value of the peaks variance
    n_max: int
        Max number of iterations
    method: string
        Optimization method: Nelder-Mead, Powell, L-BFGS-B, TNC, COBYLA, SLSQP, trust-constr

    Returns:
    flatComb_inputs0_1dB: list
        Inputs of the flatComb dataset with peaks variance less than 1 dB
    flatComb_outputs0_1dB: list
        Outputs of the flatComb dataset with peaks variance less than 1 dB
    flatComb_inputs1_ndB: list
        Inputs of the flatComb dataset with peaks variance between 1 and n dB, where n is the upper_min_max
    flatComb_outputs1_ndB: list
        Outputs of the flatComb dataset with peaks variance between 1 and n dB, where n is the upper_min_max
        
    '''
    inputs = [[urand.Uniform(low, high).sample().item() for low, high in bounds] for _ in range(nsamples)]

    flatComb_inputs0_1dB = []
    flatComb_outputs0_1dB = []

    flatComb_inputs1_ndB = []
    flatComb_outputs1_ndB = []

    for input in inputs:
        best_params, peaks, min_max = optimization_flatComb_numpy(input, args, bounds, n_max, method)
        if min_max < 1:
            flatComb_inputs0_1dB.append(best_params)
            flatComb_outputs0_1dB.append(peaks)
        elif min_max < upper_min_max:
            flatComb_inputs1_ndB.append(best_params)
            flatComb_outputs1_ndB.append(peaks)

    return flatComb_inputs0_1dB, flatComb_outputs0_1dB, flatComb_inputs1_ndB, flatComb_outputs1_ndB
