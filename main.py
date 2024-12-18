from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rpyc

app = FastAPI()


class AnswerQuestionReq(BaseModel):
    """
        Request model for the answer_question endpoint.

        Attributes:
            question (str): The user's question.
    """
    question: str


class AnswerQuestionRep(BaseModel):
    """
        Response model for the answer_question endpoint.

        Attributes:
            answer (str): The generated answer to the user's question.
    """
    answer: str


def call_answer_service(question):
    """
        Calls the external RPyC answer generation service with the given question.

        Args:
            question (str): The user's question.

        Returns:
            str: The answer generated by the RPyC service.

        Raises:
            HTTPException: If there is an error communicating with the RPyC service.
    """
    try:
        conn = rpyc.connect("localhost", 18862)
        # Call the exposed_generate_answer method
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
    """
        Endpoint to handle user questions and return generated answers.

        Args:
            AnswerQuestionReq: The request model containing the user's question.

        Returns:
            AnswerQuestionRep: The response model containing the generated answer.

        Raises:
            HTTPException: If the user's question is empty or if there is an error during processing.
    """
    if answer_question_req.question.strip() == '':
        raise HTTPException(status_code=400, detail='Question cannot be empty')
    answer = call_answer_service(answer_question_req.question)
    return AnswerQuestionRep(answer=answer)
