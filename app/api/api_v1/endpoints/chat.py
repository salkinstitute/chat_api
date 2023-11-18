from pydantic import Field
from fastapi import APIRouter

# APP related
from ..models import (
    UserMessage,
    BotMessage,
    WebsiteQuestion,
)

from ..agents.information_systems import agent as info_sys_agent

router = APIRouter(prefix="/chat", tags=["Chat"])


# Entry point, uses bot name to direct to function
@router.post("/{bot_name}")
async def chat_with_bot(payload: UserMessage, bot_name: str) -> BotMessage:
    match bot_name.lower().strip():
        case "information sytems":
            return await info_sys_agent(payload)
        case _:  # default
            return await info_sys_agent(payload)


#  Manually testing each load type first
# @router.get("")
# async def root():
#     pl = "https://www.cnn.com"
#     load_file(
#         file_type="website",
#         file_link=pl,
#         metadata_to_save=[{"source": pl, "test_data": True}],
#     )
#     return {"message": "FILE LOADED"}


# tests
