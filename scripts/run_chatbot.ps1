# Launches the Gradio chatbot with the fine-tuned model and the HotpotQA index.
#
#   .\scripts\run_chatbot.ps1                          # fine-tuned Qwen3-1.7B, hybrid + rerank
#   .\scripts\run_chatbot.ps1 -Llm hf:Qwen/Qwen3-1.7B  # base model, for comparison
#
# Then open http://127.0.0.1:7860 (first start takes ~1 minute to load the models).

param(
    [string]$EnvRoot = "D:\qmr",
    [string]$Llm = "hf:D:/qmr/runs/sft_10k/merged",
    [string]$Index = "results/full/index_n500_seed0",
    [string]$Retriever = "hybrid_rerank"
)

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:HF_HOME = Join-Path $EnvRoot "hf-cache"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONIOENCODING = "utf8"
$env:QMR_LLM = $Llm
$env:QMR_INDEX_DIR = $Index
$env:QMR_RETRIEVER = $Retriever

if (-not (Test-Path (Join-Path $Index "index.faiss"))) {
    Write-Host "No index at $Index. Build one with: python -m src.build_index --hotpotqa --n 500 --out $Index"
    exit 1
}
Write-Host "Starting chatbot (LLM $Llm, retriever $Retriever). Open http://127.0.0.1:7860 once it says 'Running on'."
& (Join-Path $EnvRoot "venv\Scripts\python.exe") -m src.app
