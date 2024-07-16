# Talking With Files.
## RAG
### What is RAG?
- RAG (Retrieval-augmented generation) is a techinique to give a LLM relavant context from external data/knowledge to answer user's query/input. This techinique generally involves Retrival of text from a database/vectorstore, prvoide the retrived indexed data to set context to LLM, then generate text relavant to user's query. 
- This techinique is helpful when you have lot of text data like documentation, pdf, terms and conditions but this whole data wouldn't fit in LLM's context window so use use RAG to make LLM answer questions on this corpous. 

### RAG pipeline
- #### Data Ingestion and Vectorstore.
    - The pipeline generally starts by collecting relavant data from database/local file/web links. It can be any text corpus which you can read programatically and convert them into text. For example you can collect bunch of video transcripts to build a RAG on.
    - Once the data is collected we try to break the large corpus of data/text into smaller chunks of text.
    - We break text using tokenizer based on chunk size which number of tokens each chunk contains.
    - Then what is a token, a token can be a word or sub-word or character, it depends on the tokenizer used for the model. So suppose you have "Hello World" text the tokens that phrase can be `Hello`, `World`. You can play around with tokenizer used by openai. [Play with tokenizer.](https://platform.openai.com/tokenizer).
    - Once we break the text into chunks/splits we convert each of this chunk into an embedding.
    - Embedding is a vector representation of some text in an embeddings space. Let me explain intutively when you here words such as embedding/embedding space, imagine there you in a room and there is a word flying at a location in the room and the coordinates of that word which is flying is the embedding and your room is the embedding space. Embedding model is something that will provide you the coordinates of the text you have in an embedding space.
    - So you use an embedding model convert each chunk of data into a embedding (a vector) and you want to have a mapping with the chunk and embedding. They can be stored in database, csv, dataframe or anything you want this storage is called vectorstore.  
- #### Retrival.
    - Now you have a vectorstore (a database of chunks of you corpous along with their embeddings mapped).
    - Next take the user's input/query and convert that peice of text into an embedding (a vector). Then we use a distance function to get a scalar number between our user's query embedding and each chunk's embedding present in our vectorstore.
    - We sort the scalar number or distance in descending order and take top K chunks to provide as context to the LLM.
    - Let see what it means, let go back to he room example I have previously talked about. Let your room be embedding space and each chunk embedding be the coordinate it is present in. So bunch of chunks are flying/lying in your room, now we are brining user's query/input into this embedding space and we know the coordinates (embedding) of the user's input.
    - Now we use distance function calculate distance between user query vector/emebdding with each of the chunked embedding and find out coordinates of the chunks (embeddings) which are near to the user query. The distance function genrally used is consine similarity. It is dot product of two vectors divided by product of thier scalar distances.
- #### Generation.
    - Now we have top K chunks of text based on the user's query.
    - We set up a system prompt as like "You are an assistant tasked to answer user queries based on the context given below. Keep it consise, precise if you don't know tell you don't know don't make up answers".
    - Then as we add context in the system prompt itself and send the user query and get the generation done by the model based on the context.
- Check the below architecture

![RAG architecture found in nvidia blogs](https://blogs.nvidia.com/wp-content/uploads/2023/11/NVIDIA-RAG-diagram-scaled.jpg)
