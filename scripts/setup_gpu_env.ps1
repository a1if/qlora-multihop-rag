# Creates the GPU training / evaluation environment on Windows.
#
#   .\scripts\setup_gpu_env.ps1                                  # defaults below
#   .\scripts\setup_gpu_env.ps1 -EnvRoot E:\qmr                  # put the env and model cache elsewhere
#
# The venv and the Hugging Face cache go under -EnvRoot (they need ~15 GB),
# so they can live on a different drive from the code. PyTorch comes from
# the CUDA 12.8 index, which RTX 50-series (Blackwell) GPUs require.

param(
    [string]$EnvRoot = "D:\qmr",
    [string]$Python = "python",
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $EnvRoot "venv"
$hfHome = Join-Path $EnvRoot "hf-cache"
$pip = Join-Path $venv "Scripts\pip.exe"
$py = Join-Path $venv "Scripts\python.exe"

New-Item -ItemType Directory -Force $hfHome | Out-Null
if (-not (Test-Path $py)) { & $Python -m venv $venv }

& $py -m pip install --no-cache-dir --upgrade pip
& $pip install --no-cache-dir torch --index-url $CudaIndex
& $pip install --no-cache-dir -r (Join-Path $repo "requirements.txt") -r (Join-Path $repo "requirements-dev.txt")

$env:HF_HOME = $hfHome
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"   # Windows without developer mode: cache copies instead of symlinks
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), '| bf16', torch.cuda.is_bf16_supported())"
& $py -c "import bitsandbytes, peft, trl, transformers; print('transformers', transformers.__version__, '| peft', peft.__version__, '| trl', trl.__version__, '| bitsandbytes', bitsandbytes.__version__)"

Write-Host ""
Write-Host "Done. In each new shell:"
Write-Host "  `$env:HF_HOME = '$hfHome'"
Write-Host "  & '$venv\Scripts\Activate.ps1'"
