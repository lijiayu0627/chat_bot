import requests
import time

API_URL = "http://localhost:8000/answer_question"


# Function to call the API
def call_api(question):
    """
        Calls the '/answer_question' API endpoint to get an answer for the user's question.

        Args:
            question (str): The user's question that needs to be answered by the chat bot.

        Returns:
            str: The generated answer to the question, or an error message if the request fails.
    """
    req = {'question': question}

    rep = requests.post(API_URL, json=req)

    if rep.status_code == 200:
        rep_data = rep.json()
        return rep_data['answer']
    else:
        return f"Error: {rep.status_code}, {rep.text}"


if __name__ == "__main__":
    # Simulate continuous API calls with user input
    while True:
        # Get user input for a question
        question = input("Enter your question: ")

        if question.lower() == 'exit':
            print("Exiting...")
            break

        # Call the API with the user's question
        answer = call_api(question)
        print(f"Answer: {answer}")


