from api_call import GPT4oClient
import json

class GPT4oEvaluator:
    def __init__(self, gpt_client):
        self.gpt_client = gpt_client

    def generate_responses(self, prompts):
        generated_answers = []
        for prompt in prompts:
            generated_response = self.gpt_client.generate_response(prompt)
            if generated_response is not None:
                generated_answers.append(generated_response)
            else:
                generated_answers.append("")
        return generated_answers

    def load_test_data(self, file_path):
        with open(file_path, "r") as f:
            test_data = [json.loads(line) for line in f]
        return test_data


if __name__ == "__main__":
    gpt_client = GPT4oClient(api_key="your-api-key-here", model="ft-your-fine-tuned-model-id")

    evaluator = GPT4oEvaluator(gpt_client)

    test_data = evaluator.load_test_data("test_data.jsonl")

    prompts = [data['prompt'] for data in test_data]

    generated_answers = evaluator.generate_responses(prompts)

    for i, generated_answer in enumerate(generated_answers):
        print(f"Test Case {i + 1}:")
        print(f"  Prompt: {prompts[i]}")
        print(f"  Generated Answer: {generated_answer}")
        print()
