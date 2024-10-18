import argparse

from slack_sdk import WebClient

from constant import SLACK_START
from utils import get_today


def send(args, templates=[]):
    sc = WebClient(token=args.slack_api_token)

    sc.chat_postMessage(
        channel=args.target_channel_name, text=SLACK_START.format(today=get_today())
    )
    for template in templates:
        result = sc.conversations_history(channel=args.target_channel_id)
        conversation_history = result["messages"]  # [0] is the most recent message
        message_ts = conversation_history[0]["ts"]

        sc.chat_postMessage(
            channel=args.target_channel_name,
            text=template,
            thread_ts=message_ts,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace_name", type=str, default="", help="Slack workspace name"
    )
    parser.add_argument(
        "--target_channel_name", type=str, default="", help="Target channel name"
    )
    parser.add_argument(
        "--target_channel_id", type=str, default="", help="Target channel ID"
    )
    parser.add_argument(
        "--slack_api_token", type=str, default="", help="Slack API token"
    )
    args = parser.parse_args()

    send(args)
