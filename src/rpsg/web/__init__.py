"""Browser front end for the arms `scripts/ask.py` asks.

`service` holds the logic and imports nothing web-specific; `app` adds FastAPI on top.
Importing `rpsg.web.app` requires the optional `web` extra, so nothing else in the package
imports it — `from rpsg.web.service import AskService` stays available either way.
"""
