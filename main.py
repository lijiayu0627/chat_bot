from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rpyc

app = FastAPI()


class AnswerQuestionReq(BaseModel):
    question: str


class AnswerQuestionRep(BaseModel):
    answer: str


def call_answer_service(question):
    try:
        conn = rpyc.connect("localhost", 18862)
        answer = conn.root.exposed_generate_answer(question)
        conn.close()
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with answer service: {str(e)}")


@app.get('/')
def read_root():
    return {'message': 'Hello, welcome to the e-commerce chat bot!'}


@app.post('/answer_question')
def answer_question(answer_question_req: AnswerQuestionReq):
    if answer_question_req.question == '':
        raise HTTPException(status_code=400, detail='Question cannot be empty')
    answer = call_answer_service(answer_question_req.question)
    return AnswerQuestionRep(answer=answer)
