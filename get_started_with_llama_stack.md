## Set up Llama Stack

After running `llama stack build`, we created a new file `csye7230-search-stack-run.yaml` under the path `/Users/YOUR_NAME/.llama/builds/docker`

```
version: v1
built_at: '2024-10-07T23:18:19.117888'
image_name: csye7230-search-stack
docker_image: csye7230-search-stack
conda_env: null
apis_to_serve:
- memory_banks
- memory
- models
- safety
- agents
- inference
- shields
api_providers:
  inference:
    providers:
    - remote::ollama
  safety:
    providers:
    - meta-reference
  agents:
    provider_type: meta-reference
    config:
      persistence_store:
        namespace: null
        type: sqlite
        db_path: /Users/friedahuang/.llama/runtime/kvstore.db
  memory:
    providers:
    - meta-reference
  telemetry:
    provider_type: meta-reference
    config: {}
routing_table:
  inference:
  - provider_type: remote::ollama
    config:
      host: localhost
      port: 11434
    routing_key: Llama3.1-8B-Instruct
  - provider_type: remote::ollama
    config:
      host: 127.0.0.1
      port: 11434
    routing_key: Llama-Guard-3-8B
  safety:
  - provider_type: meta-reference
    config:
      llama_guard_shield:
        model: Llama-Guard-3-1B
        excluded_categories: []
        disable_input_check: false
        disable_output_check: false
      enable_prompt_guard: false
    routing_key:
    - llama_guard
    - code_scanner_guard
    - injection_shield
    - jailbreak_shield
  memory:
  - provider_type: meta-reference
    config: {}
    routing_key: vector
```
