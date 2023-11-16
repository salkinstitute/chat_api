# Debugging and Logs
from pprint import pprint
import math
from urllib.parse import urlsplit, urlunsplit
import urllib.parse
from aiohttp import request
from bson.objectid import ObjectId
import motor.motor_asyncio
import os
from fastapi import Request
from datetime import datetime, timezone

from .models import *


client = motor.motor_asyncio.AsyncIOMotorClient(os.environ["MONGO_URL"])

database = client.chat_api


chat_history_collection = database.get_collection("chat_history")
datasources_collection = database.get_collection("datasources")
notes_collection = database.get_collection("notes")

User = database.users
User.create_index([("email", 1)], unique=True)

# Serializer helpers


async def serializeDict(a) -> dict:
    return {
        **{i: str(a[i]) for i in a if i == "_id"},
        **{i: a[i] for i in a if i != "_id"},
    }


async def serializeList(entity) -> list:
    return [await serializeDict(a) for a in entity]


async def userEntity(user) -> dict:
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "name": user["name"],
        "email": user["email"],
        # "rocketchat_token": user["rocketchat_token"],
        "role": user["role"],
        "photo": user["photo"],
        "verified": user["verified"],
        "password": user["password"],
        "created_at": user["created_at"],
        "updated_at": user["updated_at"],
    }


async def userResponseEntity(user) -> dict:
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "photo": user["photo"],
        "created_at": user["created_at"],
        "updated_at": user["updated_at"],
        # "rocketchat_token": user["rocketchat_token"],
    }


async def createUserResponseEntity(user) -> dict:
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "photo": user["photo"],
        "created_at": user["created_at"],
        "updated_at": user["updated_at"],
    }


async def embeddedUserResponse(user) -> dict:
    return {
        "id": str(user["_id"]),
        "name": user["name"],
        "username": user["username"],
        "email": user["email"],
        "photo": user["photo"],
    }


async def upsert_datasource(datasource: Datasource):
    ds = datasource
    now = datetime.utcnow()

    if ds.id is not None:
        ds.updated_at_utc = now
        n = await datasources_collection.update_one(
            {"_id": ds.datasource_id, "created_by": ds.updated_by},
            {"$set": ds},
        )
        return ds.datasource_id
    else:
        ds.created_at_utc = now
        n = await datasources_collection.insert_one(dict(ds))
        return n.inserted_id


async def userListEntity(users) -> list:
    return [userEntity(user) for user in users]


# get current number of users
async def retrieve_users_count() -> int:
    return await User.count_documents({})


async def retrieve_notes(question: UserNoteQuestion) -> list[UserNote] | list:
    pprint("Here are the retrieve_notes() parameters for question:")
    pprint(question)
    notes = []
    if "note_id" in question:
        # Use the note_id for the query
        q = {"_id": ObjectId(question["note_id"])}

    else:
        # Build the query based on the params given
        q = {
            "$and": [
                {"context_type": question["context_type"]},
                {"username": question["username"]},
            ]
        }
        if "context" in question:
            q["$and"].append({"context": question["context"]})

        if "title" in question:
            q["$and"].append(
                {"$or": [{"title": {"$regex": f'/{question["title"]}/i'}}]}
            )
        if "query" in question:
            q["$and"]["$or"].append({"note": {"$regex": f'/{question["query"]}/i'}})

    pprint("here is the query")
    pprint(q)
    #
    cursor = notes_collection.find(q)
    async for nt in cursor:
        nt["id"] = str(nt["_id"])
        notes.append(await serializeDict(nt))

    return notes


async def retrieve_chat_history(
    username: str,
    botname: str,
    limit: int = 5,
    sort: str = "created_at",
    sort_dir: int = -1,
):
    history = []
    cursor = (
        chat_history_collection.find(
            {"$and": [{"username": username}, {"botname": botname}]}
        )
        .limit(limit)
        .sort(sort, sort_dir)
    )
    async for h in cursor:
        h["id"] = str(h["_id"])
        history.append(await serializeDict(h))

    return history


async def retrieve_agent_datasources(
    agent: str | None = "information systems",
    limit: int = 500,
    sort: str = "created_at",
    sort_dir: int = -1,
):
    sources = []
    cursor = (
        datasources_collection.find({"ai_agent": agent})
        .limit(limit)
        .sort(sort, sort_dir)
    )
    async for h in cursor:
        h["id"] = str(h["_id"])
        sources.append(await serializeDict(h))

    # pprint(f"HERE ARE THE SOURCES {sources}")
    return sources


async def insert_chat_history(
    username: str, role: str, content: str, botname: str | None = None
):
    now = datetime.utcnow()
    doc = {
        "created_at": now,
        "username": username,
        "botname": botname,
        "role": role,
        "content": content,
    }
    n = await chat_history_collection.insert_one(doc)
    return n


async def upsert_note(
    title: str,
    username: str,
    context: str,
    context_type: str,
    note: str,
    note_id: str | None = None,
) -> str:
    now = datetime.utcnow()
    doc = {
        "title": title,
        "username": username,
        "context": context,
        "context_type": context_type,
        "note": note,
    }
    #
    if note_id is not None:
        doc["updated_at"] = now
        n = await notes_collection.update_one(
            {"note_id": note_id, "username": username}, {"$set": doc}
        )
        return note_id
    else:
        doc["created_at"] = now
        n = await notes_collection.insert_one(doc)
        return n.inserted_id


async def user_chat_history(username: str, botname: str):
    existing_messages = await retrieve_chat_history(username=username, botname=botname)
    history = ""
    for e in existing_messages:
        history += f"{e['created_at']} {e['role']} {e['content']}\n"
    print(f"Here is the history {history}")
    return history


async def get_paged_response(
    count_results: int = 0,
    page: int = 1,
    limit: int = 10,
    results: list[any] = [],
    request: Request = Request,
):
    ret = {}
    if not count_results:
        count_results = 0
    ret["total_pages"] = math.ceil(count_results / limit)
    if not ret["total_pages"]:
        ret["total_pages"] = 1
    ret["total_results"] = count_results
    next_page = page + 1 if page < ret["total_pages"] else None
    prev_page = page - 1 if page > 1 else None
    # get the url without query string or fragments
    url = urlunsplit(urlsplit(str(request.url))._replace(query="", fragment=""))
    print(f"current url and path {url}")
    # next
    if next_page:
        next_params = dict(request.query_params)
        next_params["page"] = next_page
        ret["next"] = f"{url}?{urllib.parse.urlencode(next_params)}"
    else:
        ret["next"] = ""
    # prev
    if prev_page:
        prev_params = dict(request.query_params)
        prev_params["page"] = prev_page
        ret["previous"] = f"{url}?{urllib.parse.urlencode(prev_params)}"
    else:
        ret["previous"] = ""
    ret["results"] = results

    return ret
