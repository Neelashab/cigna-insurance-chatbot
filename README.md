# Overview
Uses generative AI to intelligently scrape Cigna's website and populate a vector database. The chatbot then answers user's questions about buying health insurance using RAG. The system also includes query analysis and semantic caching, and agentic features. Deployed on serverless GCP [here][website-link]. 

[website-link]: https://frontend-cigna-chatbot-553139431985.us-east4.run.app/

# How it works

The user begins a conversation with the agent which prompts them for certain information about their business. Once it decides it has all necessary information, it moves on to searching the document database for group health insurance plans the business is eligible for. The eligible plans are then sent to another reasoning model which decides on its own metrics to rank the best one for the user's business, and then returns its reccomendations.

As I am not an insurance broker, I do not have access to structured, internal documents about group health insurance plans. To work around this, I created an automated pipeline that passes webscraped insurance plan content to a reasoning LLM that analyzes all plans collectively and decides on key metadata fields that distinguish plans from each other. These fields are sent back to a reasoning LLM along with the original, unstructured insurance plan which then populates the fields accordingly and uploads each plan document MongoDB. 

I also created a simple informative bot that answers general insurance questions using RAG with vector search. 

I avoid latency and exceeding the model's context window by pruning the initial part of conversation history intelligently with GPT-mini summaries and NLP entity extraction.

I also used GEval from ConfidentAI, previously DeepEval, to implement multi-turn, LLM-as a judge evals.

The front end repository is also publically visible, and is a simple streamlit application that is powered by the service defined in this repository. 

# Project Stack

### Data Processing & Storage
  - Beautiful Soup 4 -  HTML parsing and web scraping
  - Pydantic - Data validation
    
    **Q&A Bot**
    - LangChain - Recursive chunking for vector database 
    - Pinecone - Vector database for semantic search and RAG (Retrieval-Augmented Generation)
      
    **Sales Bot**
    - OpeanAI's GPT 4.1 Reasoning Model - Creates metadata structure and then structures plans accordingly
    - MongoDB - Document storage for generated, structured insurance plans

### Conversation Management
  - OpenAI GPT-4 - Primary chat interactions
  - OpenAI GPT-4o-mini - Lightweight model for query rewriting and intelligent conversation summaries
  - Hugging Face library with BERT for Named Entity Recognition (NER) - Aides intelligent conversation summaries

### Backend & Testing
  - FastAPI - Framework for RESTful APIs and hosting application
  - ConfidentAI's GEval - Generates conversational test cases and tests them with user provided metrics.

# Room for Improvement & Next Steps
