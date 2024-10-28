import openai
from data_processing import DataProcessor

import json

class GPT4oClient:
    def __init__(self, api_key, model="gpt-4o"):
        self.api_key = api_key
        self.model = model
        openai.api_key = self.api_key

    def generate_response(self, prompt, max_tokens=100, temperature=0.7):
        try:
            response = openai.Completion.create(
                engine=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def fine_tune_model(self, training_data, model="gpt-4o", n_epochs=3):
        try:
            file_id = self.upload_training_data(training_data)
            fine_tune_response = openai.FineTune.create(
                training_file=file_id,
                model=model,
                n_epochs=n_epochs
            )
            return fine_tune_response
        except Exception as e:
            print(f"An error occurred during fine-tuning: {e}")
            return None

    def upload_training_data(self, fine_tune_helper):
        try:
            training_data = fine_tune_helper.prepare_training_data()
            with open("training_data.jsonl", "w") as f:
                for data in training_data:
                    json.dump(data, f)
                    f.write("\n")
            response = openai.File.create(
                file=open("training_data.jsonl"),
                purpose="fine-tune"
            )
            return response["id"]
        except Exception as e:
            print(f"An error occurred while uploading the training data: {e}")
            return None

class GPTFineTuneHelper:
    def __init__(self):
        self.data_processor = DataProcessor()

    def add_question_answer(self, idx, question, answer):
        self.data_processor.add_question_answer(idx, question, answer)

    def create_finetune_prompt(self, idx):
        return self.data_processor.generate_finetune_prompt(idx)

    def prepare_training_data(self):
        training_data = []
        for idx in self.data_processor.question_data:
            prompt = self.create_finetune_prompt(idx)
            response = self.data_processor.answer_data[idx]
            training_data.append({
                "prompt": prompt,
                "completion": f" {response}"
            })
        return training_data