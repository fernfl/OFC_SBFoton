import torch

# Define the parameters class without @torch.jit.script
class parameters:
    def __init__(self, t: torch.Tensor, Rs: float, Vpi: float, P: torch.Tensor, NFFT: int, Fa: float, SpS: int, n_peaks: int):
        self.t = t
        self.Rs = Rs
        self.Vpi = Vpi
        self.P = P
        self.NFFT = NFFT
        self.Fa = Fa
        self.SpS = SpS
        self.n_peaks = n_peaks

@torch.jit.script
def analytical_function_compact_group_params(params: torch.Tensor, args: parameters) -> torch.Tensor:
    '''
    Function to generate the frequency comb signal peaks of PM-MZM-MZM using JIT
    '''
    V1, V2, V3, Phase1, Phase2, Phase3, Vb2, Vb3 = torch.split(params, 1, dim=-1)

    pi_Rs_t = 2 * torch.pi * args.Rs * args.t
    K = torch.pi / args.Vpi

    u1 = V1 * torch.cos(pi_Rs_t + Phase1)
    u2 = V2 * torch.cos(pi_Rs_t + Phase2)
    u3 = V3 * torch.cos(pi_Rs_t + Phase3)
    cos_u1_K = torch.cos(u1 * K)
    sin_u1_K = torch.sin(u1 * K)
    cos_u2_Vb2_K = torch.cos((0.5 * (u2 + Vb2)) * K)
    cos_u3_Vb3_K = torch.cos((0.5 * (u3 + Vb3)) * K)
    real_part = args.P * cos_u1_K * cos_u2_Vb2_K * cos_u3_Vb3_K
    imag_part = args.P * sin_u1_K * cos_u2_Vb2_K * cos_u3_Vb3_K
    frequency_comb = torch.complex(real_part, imag_part)

    psd = torch.fft.fft(frequency_comb, n=args.NFFT, dim=-1)
    psd = torch.fft.fftshift(psd, dim=-1)
    psd = torch.abs(psd) ** 2 / (args.NFFT * args.Fa)
    psd[psd == 0] = 1.2e-7  # torch.finfo(psd.dtype).eps
    log_Pxx = 10 * torch.log10(psd)

    n_peaks_2 = args.n_peaks // 2
    indx = (log_Pxx.size(-1) // 2) + torch.arange(-n_peaks_2, n_peaks_2 + 1, device=params.device).to(torch.int) * (args.NFFT // args.SpS)
    peaks = torch.index_select(log_Pxx, 1, indx)

    return peaks

# Example usage

device = "cpu"
params = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32).unsqueeze(0).to(device)
t = torch.arange(0, 641, dtype=torch.float32, device=device).unsqueeze(0) * 1.5625e-12
P = torch.tensor(1.0, dtype=torch.float32).unsqueeze(0).to(device)
Rs = 10e9
Vpi = 2.0
NFFT = 64
Fa = 640e9
SpS = 64
n_peaks = 9

args = parameters(t, Rs, Vpi, P, NFFT, Fa, SpS, n_peaks)

# Call the function
peaks = analytical_function_compact_group_params(params, args)
print(peaks)




