# RAG for Student Career Anxiety (Cloud Computing Project)



## What it is
For my Cloud Computing course, I created a RAG system for the topic of student career anxiety. RAG stands for retrieval-augmented generation. Instead of letting a language model answer only from its own memory, a RAG system first searches a collection of documents for the passages that match the question and then gives those passages to the language model so that it can answer based on them.

## Why I chose student career anxiety
Many students worry about their future jobs, skills and results. I wanted to build a helper that gives calm, relevant guidance on that topic. 

## How it works
1. Documents about career anxiety are collected and split into smaller chunks of text.
2. Each chunk is stored so it can be searched, for example with keyword search or with embeddings that capture meaning.
3. When a student asks a question, the system retrieves the most relevant chunks.
4. A language model reads those chunks and writes an answer based on them.


## Tools and cloud services
the language model: Gemini AI, embedding model, vector store, cloud platform and programming language you used: python 

## What I learned about RAG
Building this showed me that the quality of a RAG answer depends a lot on the documents and on how they are chunked. If the right passage is not retrieved, the model cannot answer well. It also showed me why a system should say it does not know when the documents do not contain the answer, instead of guessing. 

## Connection to this digital clone
This digital clone assignment uses the same idea. My clone retrieves passages from documents I wrote about myself, using both keyword search (BM25) and meaning-based search (embeddings), and then answers in the first person.

## Challenges
real challenges, such as finding good documents, choosing a chunk size or keeping answers on topic

## Possible improvements
better documents, evaluating answers, or adding a safety message that points to real counselors for serious problems
