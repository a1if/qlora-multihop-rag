Fine-tuning throughput · Qwen3-1.7B · 20 steps · effective batch 16 · max_len 1024

| Variant | Status | Tokens/s | Speed vs full-logits baseline | Peak VRAM (GB) |
|---|---|---|---|---|
| `lora_gc_bs4_fulllogits` | ok | 857 | 1.00× | 10.8 |
| `lora_gc_bs4` | ok | 2418 | 2.82× | 4.9 |
| `qlora_gc` | ok | 1954 | 2.28× | 5.6 |
| `lora_gc` | ok | 2158 | 2.52× | 6.0 |
| `lora_gc_bs16` | ok | 1236 | 1.44× | 9.1 |
| `lora_nogc_bs4` | out of memory | – | – | – |
| `lora_nogc_bs8` | out of memory | – | – | – |
