import argparse
import json
import time
from datasets import load_dataset
from openai import OpenAI
import random


class DialogDatabase:
    """
        Handles the preparation of training and validation data for fine-tuning GPT models
        using e-commerce customer support conversations.
    """
    def __init__(self, dataset="rjac/e-commerce-customer-support-qa"):
        """
            Initializes the DialogDatabase class by loading the dataset.

            Args:
                dataset (str): The Hugging Face dataset ID for e-commerce customer support QA.
        """
        self.ds = load_dataset(dataset, cache_dir='.')
        self.qa_conversation_dict = {}
        self.qas = []
        self.conversations = []

    def write_jsonl(self, data_list: list, filename: str) -> None:
        """
            Writes a list of dictionaries to a JSONL file.

            Args:
                data_list (list): List of dictionaries to write.
                filename (str): Name of the JSONL file.
        """
        with open(filename, "w") as out:
            for ddict in data_list:
                jout = json.dumps(ddict) + "\n"
                out.write(jout)

    def train_val_split(self, data):
        """
            Splits data into training and validation sets (80/20 split).

            Args:
                data (list): The input data to split.

            Returns:
                tuple: (training_data, validation_data)
        """
        random.shuffle(data)
        split_index = int(0.8 * len(data))
        train_data = data[:split_index]
        valid_data = data[split_index:]
        return train_data, valid_data

    def build_database(self):
        """
            Filters and processes the dataset, then generates JSONL files for
            training and validation data. Only conversations with 'neutral' or
            'positive' customer sentiment are included.
        """
        filtered_data = self.ds["train"].filter(
            lambda example: example["customer_sentiment"] in ["neutral", "positive"]
        )

        # Extract conversations and QA pairs
        result = [{"conversation": example["conversation"], "qa": example["qa"]} for example in filtered_data]
        qa_pairs = []

        # Format data for fine-tuning
        for pair in result:
            data_dict = json.loads(pair['qa'])
            if 'knowledge' in data_dict:
                for qa in data_dict['knowledge']:
                    qa_pairs.append({
                        'messages': [
                            {'role': 'system',
                             'content': 'You are a helpful assistant for a e-commerce platform. Use the following context to answer the question.'},
                            {'role': 'user', 'content': qa['customer_summary_question']},
                            {'role': 'assistant', 'content': qa['agent_summary_solution']}
                        ]
                    })

        # Split into training and validation sets
        train_data, valid_data = self.train_val_split(qa_pairs)

        # Write to JSONL files
        self.write_jsonl(train_data, "finetune_training.jsonl")
        self.write_jsonl(valid_data, "finetune_validation.jsonl")


class GPTFineTuner:
    """
        Manages fine-tuning a GPT model using prepared JSONL training and validation data.
    """
    def __init__(self, training_file="finetune_training.jsonl", valid_file="finetune_validation.jsonl"):
        """
            Initializes the GPTFineTuner class.

            Args:
                training_file (str): Path to the JSONL file with training data.
                valid_file (str): Path to the JSONL file with validation data.
        """
        self.client = OpenAI(api_key=api_key)
        self.training_file = training_file
        self.valid_file = valid_file
        self.MODEL = "gpt-4o-mini-2024-07-18"
        self.train_id, self.valid_id = self.upload_file()
        self.job_id = None

    def upload_file(self):
        """
            Uploads training and validation files to OpenAI.

            Returns:
                tuple: (training_file_id, validation_file_id)
        """
        with open(self.training_file, "rb") as file_fd:
            response = self.client.files.create(file=file_fd, purpose='fine-tune')
            train_id = response.id
        with open(self.valid_file, "rb") as file_fd:
            response = self.client.files.create(file=file_fd, purpose='fine-tune')
            valid_id = response.id
        return train_id, valid_id

    def fine_tune(self):
        """
            Starts the fine-tuning job using the uploaded training and validation files.
        """
        response = self.client.fine_tuning.jobs.create(
            training_file=self.train_id,
            validation_file=self.valid_id,
            model=self.MODEL,
            hyperparameters={'n_epochs': 1, 'batch_size': 8}
        )
        self.job_id = response.id
        print("Fine-tuning job started with ID:", self.job_id)

    def track(self):
        """
            Tracks the progress of the fine-tuning job and displays job events.
        """
        response = self.client.fine_tuning.jobs.retrieve(self.job_id)
        print("Job ID:", response.id)
        print("Status:", response.status)
        print("Trained Tokens:", response.trained_tokens)

        # Retrieve and display job events
        response = self.client.fine_tuning.jobs.list_events(self.job_id)
        events = response.data
        events.reverse()  # Display in chronological order
        for event in events:
            print(event.message)

    def save_model_id(self):
        """
            Saves the ID of the fine-tuned model to a JSON file if available.
        """
        response = self.client.fine_tuning.jobs.retrieve(self.job_id)
        if response.fine_tuned_model is not None:
            with open('fine_tuned_model_id.json', 'w') as fp:
                json.dump(response.fine_tuned_model, fp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fine Tune GPT")
    parser.add_argument('-k', '--key', type=str, required=True, help='Your OpenAI API Key')
    args = parser.parse_args()
    api_key = args.key

    # Initialize the dialog database and prepare training/validation data
    data = DialogDatabase()
    data.build_database()

    # Initialize the fine-tuner and start the fine-tuning process
    gpt = GPTFineTuner()
    gpt.fine_tune()

    # Wait for the fine-tuning job to complete (replace with appropriate tracking in real use cases)
    time.sleep(2000)

    # Track fine-tuning progress and save the model ID
    gpt.track()
    gpt.save_model_id()