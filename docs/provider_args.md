# Provider Args

## QwenLocalProvider
- `temperature` (float): Sampling temperature for text generation.
- `max_new_tokens` (int, default: 512): Maximum tokens generated per reply.

## DeepSeekProvider
- `temperature` (float): Sampling temperature for text generation.
- `max_new_tokens` (int, default: 512): Maximum tokens generated per reply.
- `model` (str, default: `deepseek-chat`): Model ID when using the API.
- `model_path` (str, optional): Local checkpoint path for offline usage.
- `api_key` (str, optional): API key for DeepSeek's hosted service.
