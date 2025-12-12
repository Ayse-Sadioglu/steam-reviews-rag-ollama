# Steam Reviews RAG with Ollama

This project is a Retrieval Augmented Generation (RAG) assistant that answers questions about Steam game reviews using a local LLaMA model via Ollama and FAISS vector search. The system generates answers strictly based on player reviews.

The dataset used in this project is the Steam Reviews dataset from Kaggle:
https://www.kaggle.com/datasets/andrewmvd/steam-reviews

To use the dataset, download it from Kaggle and place the CSV file under the `data/` directory as `steam_reviews.csv`.

The assistant loads review texts, creates vector embeddings, retrieves relevant reviews, and generates grounded responses using RAG.

Technologies used include Python, LangChain, FAISS, Ollama (LLaMA), Sentence Transformers, and Pandas.
