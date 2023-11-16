import asyncio
import os
from pprint import pprint
import requests
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
import nest_asyncio

# APP the agent to use (same as API based)
from ..api.api_v1.agents.information_systems import agent as is_agent
from ..api.api_v1.models import UserMessage


nest_asyncio.apply()


async def chat_slack():
    if os.environ.get("SLACK_BOT_TOKEN") and os.environ.get("SLACK_APP_TOKEN"):
        app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))
        response = requests.post(
            "https://slack.com/api/auth.test",
            headers={"Authorization": "Bearer " + os.environ.get("SLACK_BOT_TOKEN")},
        )
        bot_user_id = response.json()["user_id"]
        print(f"Pinged slack, botname is {bot_user_id}")

        # @app.message(f"<@{bot_user_id}>")
        @app.message("")
        async def message_all(message, say):
            print("Message received... ")
            # message from the user
            pprint(message)
            user_input = message["text"]
            # user's id, used for replies and lookups
            uid = message["user"]
            # request full user info
            user_req = await app.client.users_info(user=uid)
            pprint(user_req)
            # parse
            user = user_req.data["user"]
            pprint(user)
            # quick spinner during update, do a message change afterwards
            # works but ugly
            # thinking = await app.client.files_upload(
            #     channels=message["channel"],
            #     initial_comment=":thinking_face:",
            #     file="/app/app/slack/thinking.gif",
            # )
            # emoji is faster
            thinking = await say(
                text=f"<@{uid}> :thinking_face:",
                # mrkdwn=True,
            )

            # pprint(thinking)

            # use giphy instead, nope slash commands don't work
            # if really want to waste time, could use the giphy api
            # await say(text=f"<@{uid}> /giphy thinking")

            print(f"Incoming message from {user}")

            user_message = UserMessage(
                **{
                    "context": "slackbot-is",
                    "contextType": "channel",
                    "username": user["name"],
                    "message": message["text"],
                }
            )
            print(f"Here is the dict for sending to the Agent:\n{user_message}")
            # response = asyncio.run(slackbot_agent(user_message))
            response = asyncio.run(is_agent(user_message))

            await say(
                text=f"<@{uid}> {response['message']}",
                # mrkdwn=True,
            )

        socket_handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        await socket_handler.start_async()
    else:
        print(
            "SLACK_BOT_TOKEN and SLACK_APP_TOKEN environment variables are not found. Exiting..."
        )


async def run():
    await chat_slack()


def main():
    asyncio.run(run())


main()
