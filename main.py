from typing import Union

from fastapi import FastAPI

from tflite_support.task import text

app = FastAPI()


@app.get("/")
def get_intent(q: Union[str, None] = None):

    # initialize the task searcher
    text_searcher = text.TextSearcher.create_from_file('CliveTFLiteIntentSearcherModel00.tflite')

    # run inference
    result = text_searcher.search(q)
    intent = int(result.nearest_neighbors[0].metadata.decode())
    confidence = round(abs(result.nearest_neighbors[0].distance * 100))

    return {"user prompt": q,
            "intent": intent,
            "confidence": confidence}

