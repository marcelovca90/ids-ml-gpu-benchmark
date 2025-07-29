from datetime import datetime
from discord_webhook import DiscordWebhook

IS_ENABLED = False
WEBHOOK_URL = "INSERT_DISCORD_WEBHOOK_URL_HERE"

def now():
    now = datetime.now()
    yyyymmdd_hhmmss_part = now.strftime('%Y-%m-%d %H:%M:%S')
    ms_part = f'{int(now.microsecond / 1000):03d}'
    return f'{yyyymmdd_hhmmss_part},{ms_part}'

def post_disc(content):
    response = None
    if IS_ENABLED:
        try:
            webhook = DiscordWebhook(WEBHOOK_URL, content=f'[{now()}] {content}')
            response = webhook.execute()
        except Exception as e:
            response = f'Error: {str(e)}'
    return response

if __name__ == "__main__":
    post_disc("hello world")