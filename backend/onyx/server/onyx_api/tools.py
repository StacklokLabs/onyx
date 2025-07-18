from fastapi import APIRouter
from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.auth.users import api_key_dep
from onyx.chat.models import AnswerStyleConfig
from onyx.chat.models import CitationConfig
from onyx.chat.models import DocumentPruningConfig
from onyx.chat.models import PromptConfig
from onyx.context.search.enums import LLMEvaluationType
from onyx.context.search.models import RetrievalDetails
from onyx.db.engine.sql_engine import get_session
from onyx.db.models import User
from onyx.db.persona import get_persona_by_id
from onyx.prompts.prompt_utils import build_complete_context_str
from onyx.llm.factory import get_default_llms
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool_implementations.search_like_tool_utils import (
    FINAL_CONTEXT_DOCUMENTS_ID,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()

router = APIRouter(prefix="/onyx-tools")


class SearchToolRequest(BaseModel):
    query: str


class DocumentResult(BaseModel):
    document_id: str
    title: str
    content: str
    source_type: str
    link: str | None
    source_links: dict[int, str] | None
    metadata: dict[str, str | list[str]]
    updated_at: str | None


class SearchToolResponse(BaseModel):
    results: str


@router.post("/search-tool")
def search_tool_endpoint(
    request: SearchToolRequest,
    _: User | None = Depends(api_key_dep),
    db_session: Session = Depends(get_session),
) -> SearchToolResponse:
    """
    Endpoint that exposes the SearchTool.run() method for MCP server integration.

    This endpoint initializes a SearchTool instance and runs a search query,
    returning the final context documents as structured results.
    """
    logger.info(f"Received SearchTool request with query: {request.query}")

    # Get default LLMs
    primary_llm, fast_llm = get_default_llms()

    # Get default persona (id=0)
    persona = get_persona_by_id(0, None, db_session)

    # Set up configurations
    retrieval_options = RetrievalDetails()
    prompt_config = PromptConfig.from_model(persona.prompts[0])
    pruning_config = DocumentPruningConfig()
    answer_style_config = AnswerStyleConfig(citation_config=CitationConfig())
    evaluation_type = LLMEvaluationType.SKIP

    # Create SearchTool instance
    search_tool = SearchTool(
        db_session=db_session,
        user=None,  # No specific user
        persona=persona,
        retrieval_options=retrieval_options,
        prompt_config=prompt_config,
        llm=primary_llm,
        fast_llm=fast_llm,
        document_pruning_config=pruning_config,
        answer_style_config=answer_style_config,
        evaluation_type=evaluation_type,
    )

    # Run the search
    results = []
    try:
        for response in search_tool.run(query=request.query):
            results.append(response)

        # Extract the final context documents
        final_docs_response = next((response for response in results if response.id == FINAL_CONTEXT_DOCUMENTS_ID), None)

        if final_docs_response:
            # Extract document information for structured response
            for doc in final_docs_response.response:
                if doc.metadata:
                    doc.metadata["link"] = doc.link
                else:
                    doc.metadata = {"link": doc.link}

            final_docs_str = build_complete_context_str(final_docs_response.response)

            return SearchToolResponse(results=final_docs_str)
        else:
            logger.warning("No final context documents found in search results")
            return SearchToolResponse(results="")

    except Exception as e:
        logger.error(f"Error running SearchTool: {str(e)}", exc_info=True)
        return SearchToolResponse(results="")
