from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union


class AddSource(BaseModel):
    file_link: str = Field(description="The link to the data source you wish to add")
    file_type: str = Field(
        description="The type of data source you wish to add. Currently the choices are Doc, Pdf or Website"
    )
    title: Union[str, None] = Field(
        description="A 4 word or less sparse summary of what this data is and what it can be used for, easy to remember for the user.  If the file_type is Csv this is very useful and you should ask the user for the title before continuing.",
        default=None,
    )


class Datasource(BaseModel):
    id: Union[str, None] = Field(
        description="The mongo ObjectId from this document in the collection",
        default=None,
        alias="_id",
    )
    ai_agent: str = Field(
        description="The name of the AI Agent who added (and has implied access to) this datasource"
    )
    file_link: str = Field(description="The link to the data source you wish to add")
    file_type: str = Field(
        description="The type of data source you wish to add. Currently the choices are Doc, Pdf or Website"
    )
    title: Union[str, None] = Field(
        description="A 4 word or less sparse summary of what this data is and what it can be used for, easy to remember for the user.  If the file_type is Csv this is very useful and you should ask the user for the title before continuing.",
        default=None,
    )
    created_at_utc: Union[str, None] = Field(
        description="UTC created time", default=None
    )
    updated_at_utc: Union[str, None] = Field(
        description="UTC updated time", default=None
    )
    created_by: Union[str, None] = Field(
        description="The username who requested to have the source added", default=None
    )
    updated_by: Union[str, None] = Field(
        description="username of updator", default=None
    )
    sparse_summary: Union[str, None] = Field(
        description="A 40 word or less sparse summary of what this data is and what it can be used for",
        default=None,
    )
    s3_key: Union[str, None] = Field(description="The link to the file in AWS s3 in s3://filename style", default=None)


class UserMessage(BaseModel):
    context: Any
    contextType: str
    username: str
    message: str


class BotMessage(BaseModel):
    username: str
    context: Any
    contextType: str
    message: Any
    sources: List[Dict] | None


class ChatQuestion(BaseModel):
    username: str
    question: str


class ChatAnswer(BaseModel):
    username: str
    answer: str
    sources: List[Dict] | None


class WebsiteQuestion(BaseModel):
    website_urls: list[str] = Field(
        description="A list of website urls to use as sources for question and answer"
    )
    question: str = Field(
        description="The question the user to research in the webiste data"
    )


class UserNoteQuestion(BaseModel):
    query: str | None = Field(
        description="Words or topic to search for in the notes", default=None
    )
    title: str | None = Field(
        description="A 12 word or less title based on a sparse summary of the note.",
        default=None,
    )
    context: str = Field(
        description="The id or unique name of the entity for which the 'context_type' of the note is associated with.  For example, when the context_type is 'Bill' the context will be the bill_slug"
    )
    context_type: str = Field(
        description="The entity the note is about, for example a Bill"
    )
    username: str = Field(
        description="The name of the user(politician) whoose note it is"
    )
    note_id: str | None = Field(
        description="The note_id is the unique id for the note in the database, this is the most accurate means of retreiving a specific note when possible.",
        default=None,
    )


class UserNote(BaseModel):
    note: str = Field(
        description="The content the user has asked you to save, escaped it a json format for inserting into a mongo document"
    )
    title: str = Field(
        description="A 12 word or less title based on a sparse summary of the note."
    )
    context: str = Field(
        description="The id or unique name of the entity for which the 'context_type' of the note is associated with.  For example, when the context_type is 'Bill' the context will be the bill_slug"
    )
    context_type: str = Field(
        description="The entity the note is about, for example a Bill"
    )
    username: str = Field(
        description="The name of the user(politician) who asks for the note"
    )
    note_id: str | None = Field(
        description="If the user wants to edit an existing note, then the note_id will already exist and be available somewhere in the context, otherwise omitt this parameter (Default is None)",
        default=None,
    )
