import re

class DataProcessor:
    def __init__(self):
        self.question_data = {}
        self.answer_data = {}
        self.product_data = {}
        self.product_attributes = {}
        self.product_description = {}
        self.product_wiki_data = {}

    def add_question_answer(self, idx, question, answer):
        self.question_data[idx] = question
        self.answer_data[idx] = answer

    def add_product_info(self, idx, product_name, attributes, description):
        self.product_data[idx] = product_name
        self.product_attributes[product_name] = self.extract_product_info_from_text(description,attributes)
        self.product_description[product_name] = description

    def add_product_wiki_data(self, product_name, wiki_info):
        self.product_wiki_data[product_name] = wiki_info

    def extract_product_info_from_text(self, text, attributes):
        product_info = {}
        for attribute in attributes:
            pattern = rf'{attribute}: (\S+)'
            match = re.search(pattern, text)
            if match:
                product_info[attribute] = match.group(1)
        return product_info

    def generate_finetune_prompt(self, idx):
        question = self.question_data.get(idx)
        answer = self.answer_data.get(idx)

        if not question or not answer:
            raise ValueError(f"No data found for index {idx}")

        prompt = (f"You are a helpful customer service representative.\n"
                  f"Customer's Question: {question}\n"
                  f"Your Response: {answer}\n")
        return prompt
