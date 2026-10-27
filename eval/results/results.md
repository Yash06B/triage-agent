# Results

| Model | Strategy | n | Type acc. | Dup recall | Dup precision | False-dup rate | Area P / R | Areas rejected | Tool calls | p50 latency | Tokens in / out | Forced submit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama/llama3.2:3b | zero-shot | 120 | 73.9% | 0.0% | n/a | 0.0% | 4.7% / 66.7% | 11.7% | 0 | 1.4 s | 1,063 / 86 | 0.0% |
| ollama/llama3.2:3b | rag | 120 | 75.7% | 2.5% | 50.0% | 1.2% | 5.1% / 77.8% | 5.8% | 0 | 1.7 s | 1,398 / 106 | 0.0% |
| ollama/llama3.2:3b | agent | 120 | 51.3% | 0.0% | 0.0% | 18.8% | 5.9% / 55.6% | 45.8% | 2.69 | 10.2 s | 10,697 / 613 | 66.7% |
| ollama/qwen3:4b-instruct | zero-shot | 120 | 76.5% | 0.0% | n/a | 0.0% | 6.8% / 88.9% | 0.8% | 0 | 3.3 s | 1,081 / 171 | 0.0% |
| ollama/qwen3:4b-instruct | rag | 120 | 76.5% | 15.0% | 37.5% | 8.8% | 5.6% / 77.8% | 0.8% | 0 | 3.7 s | 1,436 / 194 | 0.0% |
| ollama/qwen3:4b-instruct | agent | 120 | 72.2% | 22.5% | 22.5% | 27.5% | 5.4% / 88.9% | 0.0% | 3.37 | 9.1 s | 10,021 / 472 | 53.3% |

Title search finds the true duplicate in the top 5 for 13/40 duplicates (32%).

## v1 agent (before the fixes described in the README)

| Model | Strategy | n | Type acc. | Dup recall | Dup precision | False-dup rate | Area P / R | Areas rejected | Tool calls | p50 latency | Tokens in / out | Forced submit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama/llama3.2:3b | agent | 120 | 53.9% | 0.0% | 0.0% | 3.8% | 4.7% / 33.3% | 32.5% | 1.13 | 1.9 s | 3,296 / 205 | 11.7% |
