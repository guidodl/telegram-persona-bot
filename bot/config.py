from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    telegram_bot_token: str = ""
    bot_name: str = ""  # name the bot answers to in groups when addressed without an @mention
    bot_aliases: str = ""  # comma-separated extra names the bot answers to in groups
    group_allowed_chats: str = ""  # comma-separated group chat IDs; empty = respond in any group
    group_allowed_topics: str = ""  # comma-separated chat_id:message_thread_id pairs; a chat with no entry is unrestricted
    allowed_users: str = ""  # comma-separated Telegram user IDs allowed to DM the bot; empty = everyone
    persona_file: str = ""  # path to a persona definition file; empty = generic companion persona
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    tavily_api_key: str = ""
    brave_api_key: str = ""
    database_url: str = ""
    model_chat: str = "deepseek/deepseek-v4-flash"
    model_agent: str = ""  # model for the tool-loop decisions (e.g. whether to send a pic); empty = same as model_chat
    model_vision: str = "google/gemini-2.5-flash"
    model_embed: str = "openai/text-embedding-3-small"
    tts_model: str = "fish-audio/s2.1-pro-free"  # OpenRouter speech model; see ?output_modalities=speech
    tts_voice: str = ""  # must be a supported_voices entry for tts_model; empty = provider default
    embed_dim: int = 1536
    embed_backend: str = "openrouter"  # or "openai"
    pacing_enabled: bool = True
    pacing_delay_min_s: float = 0.5
    pacing_delay_max_s: float = 2.0
    agent_max_iterations: int = 6  # bounds the hand-rolled tool loop; each iteration is a serial LLM round-trip, so this caps worst-case reply latency
    hermes_url: str = "http://hermes:8642"
    hermes_api_key: str = ""  # must equal the sidecar's API_SERVER_KEY; sent as Bearer to Hermes' /v1 api_server adapter
    mcp_tools_url: str = "http://mcp-tools:8000"


settings = Settings()
