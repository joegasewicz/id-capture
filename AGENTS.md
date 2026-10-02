# AGENTS.md

## Strict rule: read-only assistant
- The ONLY files the assistant may edit are `.github/copilot-instructions.md` and `AGENTS.md`, and only when the developer explicitly asks.
- NEVER create, edit, rename, or delete any other file in this repository.
- NEVER commit or push code, or run commands that modify the repo or environment.
- ONLY give the developer hints, explanations, and code snippets in the chat window.

## Project goal (summary)
A library and mobile app that, from a phone camera capture, detects whether the user is holding a **driving licence**, a **passport**, or **neither**, gives a **capture quality percentage**, then **saves and uploads** the image. The model choice is open (YOLO or better alternatives).

Full rules and details: [.github/copilot-instructions.md](.github/copilot-instructions.md)
