from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import ask_agent

app = FastAPI()


class AgentRequest(BaseModel):
    question: str


@app.post("/agent")
def run_agent(request: AgentRequest):
    try:
        answer = ask_agent(request.question)

        return {
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )