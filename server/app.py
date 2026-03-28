from __future__ import annotations

import uvicorn

from app.env import app


def main() -> None:
    uvicorn.run("app.env:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
