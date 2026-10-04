# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "vaderSentiment",
# ]
# ///

from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()

# Let any website call this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The "brain" that scores text
analyzer = SentimentIntensityAnalyzer()


# Describes what the incoming JSON must look like
class SentimentRequest(BaseModel):
    sentences: List[str]


def classify(sentence: str) -> str:
    # compound score runs from -1 (very negative) to +1 (very positive)
    score = analyzer.polarity_scores(sentence)["compound"]
    if score >= 0.05:
        return "happy"
    if score <= -0.05:
        return "sad"
    return "neutral"


@app.post("/sentiment")
async def sentiment(req: SentimentRequest):
    results = [
        {"sentence": s, "sentiment": classify(s)}
        for s in req.sentences
    ]
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)