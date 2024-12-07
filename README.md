# E-commerce Customer Service Chatbot
## Motivation
E-commerce businesses often struggle with handling repetitive customer inquiries, which can be time-consuming and impact the customer experiences. 

To solve this, I propose an AI-powered assistant designed for e-commerce, providing quick and accurate responses to improve customer service efficiency.

## Overview
This project is designed to create a domain-specific customer service chatbot that utilizes Retrieval-Augmented Generation (RAG) 
to accurately respond to customer inquiries. The chatbot aims to assist users with a variety of e-commerce-related questions, 
including but not limited to product details, shipping status, payment options, return policies, and more.

There are two knowledge databases, product database and FAQ database. 
The product database contains detailed information about the products sold by the store, such as object, category, price and delivery date.
The product information is made up here, and you can replace it with the real information of products your store is selling.
The FAQ database includes a comprehensive set of frequently asked questions and store policies, 
sourced from a publicly available dataset on **[Kaggle](https://www.kaggle.com/datasets/saadmakhdoom/ecommerce-faq-chatbot-dataset)**. 

When queried, the chatbot will retrieve related knowledge and FAQs from the existing databases as a reference and answer the query based on that information.
The chatbot can deliver contextually accurate answers based on the conversation history.

The chatbot can be fine-tuned based on a customer service **[dataset](https://huggingface.co/datasets/rjac/e-commerce-customer-support-qa)**. 
We select conversations with positive feedback as fine-tuning examples to enhance the chatbot's capability to serve customers effectively.

By combining retrieval from the knowledge base with the fine-tuned language model, the chatbot can deliver accurate and user-friendly answers.

## Technical Approach
* __Retrieval-Augmented Generation (RAG)__: to fetch accurate product and FAQ information and to provide domain-specific and personalized customer service.
* __Sentence-BERT__: to process and encode product information and FAQ information.
* __FAISS (Facebook AI Similarity Search)__: to store embeddings for efficient information retrieval enabling faster responses.
* __Finetuned GPT-4o-mini__: to generate accurate, context-aware, and relevant responses tailored to the e-commerce context.
* __FastAPI__: to build RESTful API for handling a large volume of requests from customers with minimal latency.
* __RPC (Remote Procedure Call)__: to facilitate communication between the web server with the answer generation service with low latency and high reliability.

## Running Guidline

### Fine-tune GPT
```buildoutcfg
python finetune_GPT.py -k <Your OpenAI API Key>
```
The ID of the fine-tuned model would be writen into fine_tuned_model_id.json.

If you skip this fine tune step, the answer generation service would still work with gpt-4o-mini-2024-07-18.

### Run Answer Generation Service
```buildoutcfg
python answer_generation.py -k <Your OpenAI API Key>
```

### Start Chatbot Web Server
```buildoutcfg
uvicorn main:app --reload
```