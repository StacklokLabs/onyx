from typing import cast
from uuid import UUID

from pydantic import BaseModel
from pydantic import Field
from sqlalchemy.orm import Session

from onyx.chat.models import AnswerStyleConfig
from onyx.chat.models import CitationConfig
from onyx.chat.models import DocumentPruningConfig
from onyx.chat.models import PromptConfig
from onyx.configs.app_configs import AZURE_DALLE_API_BASE
from onyx.configs.app_configs import AZURE_DALLE_API_KEY
from onyx.configs.app_configs import AZURE_DALLE_API_VERSION
from onyx.configs.app_configs import AZURE_DALLE_DEPLOYMENT_NAME
from onyx.configs.app_configs import IMAGE_MODEL_NAME
from onyx.configs.model_configs import GEN_AI_TEMPERATURE
from onyx.context.search.enums import LLMEvaluationType
from onyx.context.search.enums import OptionalSearchSetting
from onyx.context.search.models import InferenceSection
from onyx.context.search.models import RerankingDetails
from onyx.context.search.models import RetrievalDetails
from onyx.db.llm import fetch_existing_llm_providers
from onyx.db.models import Persona
from onyx.db.models import User
from onyx.file_store.models import InMemoryChatFile
from onyx.llm.interfaces import LLM
from onyx.llm.interfaces import LLMConfig
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.tools.built_in_tools import get_built_in_tool_by_id
from onyx.tools.models import DynamicSchemaInfo
from onyx.tools.tool import Tool
from onyx.tools.tool_implementations.custom.custom_tool import (
    build_custom_tools_from_openapi_schema_and_headers,
)
from onyx.tools.tool_implementations.images.image_generation_tool import (
    ImageGenerationTool,
)
from onyx.tools.tool_implementations.internet_search.internet_search_tool import (
    InternetSearchTool,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.utils import compute_all_tool_tokens
from onyx.tools.utils import explicit_tool_calling_supported
from onyx.utils.headers import header_dict_to_header_list
from onyx.utils.logger import setup_logger

logger = setup_logger()


class SearchToolConfig(BaseModel):
    answer_style_config: AnswerStyleConfig = Field(
        default_factory=lambda: AnswerStyleConfig(citation_config=CitationConfig())
    )
    document_pruning_config: DocumentPruningConfig = Field(
        default_factory=DocumentPruningConfig
    )
    retrieval_options: RetrievalDetails = Field(default_factory=RetrievalDetails)
    rerank_settings: RerankingDetails | None = None
    selected_sections: list[InferenceSection] | None = None
    chunks_above: int = 0
    chunks_below: int = 0
    full_doc: bool = False
    latest_query_files: list[InMemoryChatFile] | None = None
    # Use with care, should only be used for OnyxBot in channels with multiple users
    bypass_acl: bool = False


class InternetSearchToolConfig(BaseModel):
    answer_style_config: AnswerStyleConfig = Field(
        default_factory=lambda: AnswerStyleConfig(
            citation_config=CitationConfig(all_docs_useful=True)
        )
    )
    document_pruning_config: DocumentPruningConfig = Field(
        default_factory=DocumentPruningConfig
    )


class ImageGenerationToolConfig(BaseModel):
    additional_headers: dict[str, str] | None = None


class CustomToolConfig(BaseModel):
    chat_session_id: UUID | None = None
    message_id: int | None = None
    additional_headers: dict[str, str] | None = None


def _get_image_generation_config(llm: LLM, db_session: Session) -> LLMConfig:
    """Helper function to get image generation LLM config based on available providers"""
    if llm and llm.config.api_key and llm.config.model_provider == "openai":
        return LLMConfig(
            model_provider=llm.config.model_provider,
            model_name=IMAGE_MODEL_NAME,
            temperature=GEN_AI_TEMPERATURE,
            api_key=llm.config.api_key,
            api_base=llm.config.api_base,
            api_version=llm.config.api_version,
            max_input_tokens=llm.config.max_input_tokens,
        )

    if llm.config.model_provider == "azure" and AZURE_DALLE_API_KEY is not None:
        return LLMConfig(
            model_provider="azure",
            model_name=f"azure/{AZURE_DALLE_DEPLOYMENT_NAME}",
            temperature=GEN_AI_TEMPERATURE,
            api_key=AZURE_DALLE_API_KEY,
            api_base=AZURE_DALLE_API_BASE,
            api_version=AZURE_DALLE_API_VERSION,
            max_input_tokens=llm.config.max_input_tokens,
        )

    # Fallback to checking for OpenAI provider in database
    llm_providers = fetch_existing_llm_providers(db_session)
    openai_provider = next(
        iter(
            [
                llm_provider
                for llm_provider in llm_providers
                if llm_provider.provider == "openai"
            ]
        ),
        None,
    )

    if not openai_provider or not openai_provider.api_key:
        raise ValueError("Image generation tool requires an OpenAI API key")

    return LLMConfig(
        model_provider=openai_provider.provider,
        model_name=IMAGE_MODEL_NAME,
        temperature=GEN_AI_TEMPERATURE,
        api_key=openai_provider.api_key,
        api_base=openai_provider.api_base,
        api_version=openai_provider.api_version,
        max_input_tokens=llm.config.max_input_tokens,
    )


# Note: this is not very clear / not the way things should generally be done. (+impure function)
# TODO: refactor the tool config flow to be easier
def _configure_document_pruning_for_tool_config(
    tool_config: SearchToolConfig | InternetSearchToolConfig,
    tools: list[Tool],
    llm: LLM,
) -> None:
    """Helper function to configure document pruning settings for tool configs"""
    tool_config.document_pruning_config.tool_num_tokens = compute_all_tool_tokens(
        tools,
        get_tokenizer(
            model_name=llm.config.model_name,
            provider_type=llm.config.model_provider,
        ),
    )
    tool_config.document_pruning_config.using_tool_message = (
        explicit_tool_calling_supported(
            llm.config.model_provider, llm.config.model_name
        )
    )


def construct_tools(
    persona: Persona,
    prompt_config: PromptConfig,
    db_session: Session,
    user: User | None,
    llm: LLM,
    fast_llm: LLM,
    run_search_setting: OptionalSearchSetting,
    search_tool_config: SearchToolConfig | None = None,
    internet_search_tool_config: InternetSearchToolConfig | None = None,
    image_generation_tool_config: ImageGenerationToolConfig | None = None,
    custom_tool_config: CustomToolConfig | None = None,
) -> dict[int, list[Tool]]:
    """Constructs tools based on persona configuration and available APIs"""
    tool_dict: dict[int, list[Tool]] = {}

    # Get user's OAuth token if available
    user_oauth_token = None
    if user and user.oauth_accounts:
        user_oauth_token = user.oauth_accounts[0].access_token

    for db_tool_model in persona.tools:
        if db_tool_model.in_code_tool_id:
            tool_cls = get_built_in_tool_by_id(
                db_tool_model.in_code_tool_id, db_session
            )

            # Handle Search Tool
            if (
                tool_cls.__name__ == SearchTool.__name__
                and run_search_setting != OptionalSearchSetting.NEVER
            ):
                if not search_tool_config:
                    search_tool_config = SearchToolConfig()

                search_tool = SearchTool(
                    db_session=db_session,
                    user=user,
                    persona=persona,
                    retrieval_options=search_tool_config.retrieval_options,
                    prompt_config=prompt_config,
                    llm=llm,
                    fast_llm=fast_llm,
                    document_pruning_config=search_tool_config.document_pruning_config,
                    answer_style_config=search_tool_config.answer_style_config,
                    selected_sections=search_tool_config.selected_sections,
                    chunks_above=search_tool_config.chunks_above,
                    chunks_below=search_tool_config.chunks_below,
                    full_doc=search_tool_config.full_doc,
                    evaluation_type=(
                        LLMEvaluationType.BASIC
                        if persona.llm_relevance_filter
                        else LLMEvaluationType.SKIP
                    ),
                    rerank_settings=search_tool_config.rerank_settings,
                    bypass_acl=search_tool_config.bypass_acl,
                )
                tool_dict[db_tool_model.id] = [search_tool]

            # Handle Image Generation Tool
            elif tool_cls.__name__ == ImageGenerationTool.__name__:
                if not image_generation_tool_config:
                    image_generation_tool_config = ImageGenerationToolConfig()

                img_generation_llm_config = _get_image_generation_config(
                    llm, db_session
                )

                tool_dict[db_tool_model.id] = [
                    ImageGenerationTool(
                        api_key=cast(str, img_generation_llm_config.api_key),
                        api_base=img_generation_llm_config.api_base,
                        api_version=img_generation_llm_config.api_version,
                        additional_headers=image_generation_tool_config.additional_headers,
                        model=img_generation_llm_config.model_name,
                    )
                ]

            # Handle Internet Search Tool
            elif tool_cls.__name__ == InternetSearchTool.__name__:
                if not internet_search_tool_config:
                    internet_search_tool_config = InternetSearchToolConfig()

                try:
                    tool_dict[db_tool_model.id] = [
                        InternetSearchTool(
                            db_session=db_session,
                            persona=persona,
                            prompt_config=prompt_config,
                            llm=llm,
                            document_pruning_config=internet_search_tool_config.document_pruning_config,
                            answer_style_config=internet_search_tool_config.answer_style_config,
                            provider=None,  # Will use default provider
                            num_results=10,
                        )
                    ]
                except ValueError as e:
                    logger.error(f"Failed to initialize Internet Search Tool: {e}")
                    raise ValueError(
                        "Internet search tool requires a Bing or Exa API key, please contact your Onyx admin to get it added!"
                    )

        # Handle custom tools
        elif db_tool_model.openapi_schema:
            if not custom_tool_config:
                custom_tool_config = CustomToolConfig()

            tool_dict[db_tool_model.id] = cast(
                list[Tool],
                build_custom_tools_from_openapi_schema_and_headers(
                    db_tool_model.openapi_schema,
                    dynamic_schema_info=DynamicSchemaInfo(
                        chat_session_id=custom_tool_config.chat_session_id,
                        message_id=custom_tool_config.message_id,
                    ),
                    custom_headers=(db_tool_model.custom_headers or [])
                    + (
                        header_dict_to_header_list(
                            custom_tool_config.additional_headers or {}
                        )
                    ),
                    user_oauth_token=(
                        user_oauth_token if db_tool_model.passthrough_auth else None
                    ),
                ),
            )

    tools: list[Tool] = []
    for tool_list in tool_dict.values():
        tools.extend(tool_list)

    # factor in tool definition size when pruning
    if search_tool_config:
        _configure_document_pruning_for_tool_config(search_tool_config, tools, llm)

    if internet_search_tool_config:
        _configure_document_pruning_for_tool_config(
            internet_search_tool_config, tools, llm
        )

    return tool_dict
