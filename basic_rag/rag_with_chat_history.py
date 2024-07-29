import openai
import torch
import os
from pdfminer.high_level import extract_text
import tiktoken
from dotenv import load_dotenv

# load you openai key from .env file
load_dotenv()

CHUNK_SIZE = 512  # The target size of each text chunk in tokens
MIN_CHUNK_SIZE_CHARS = 350  # The minimum size of each text chunk in characters
MIN_CHUNK_LENGTH_TO_EMBED = 5  # Discard chunks shorter than this
EMBEDDINGS_BATCH_SIZE = int(
    os.environ.get("OPENAI_EMBEDDING_BATCH_SIZE", 128)
)  # The number of embeddings to request at a time
MAX_NUM_CHUNKS = 10000
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOKENIZER_MODEL = "o200k_base"

# Lets start by getting data
def get_data():
    """
    Provide the files you want to talk to.
    """
    while True:
        file_paths = input("Enter path of the pdf:  ")
        if (
            os.path.exists(file_paths) and os.path.splitext(file_paths)[-1] == ".pdf"
        ):  # For now we are expecting only a single file you can change this behaviour further
            break
        print("Enter the correct file path and try again", end="\r")
    return file_paths


## read the files and and chunk text
def get_text_chunk(
    file_path: str,
    chunk_size: int = CHUNK_SIZE,
):
    """
    Implementation of text splitter based on fixed chunk size
    """
    tokenizer = tiktoken.get_encoding(TOKENIZER_MODEL)
    # convert pdf to text.
    text = extract_text(file_path)
    chunks = []
    # get tokens from text
    tokens = tokenizer.encode(text, disallowed_special=())
    num_chunks = 0
    
    # Loop until tokens are consumed
    while tokens and num_chunks < MAX_NUM_CHUNKS:

        # lets start taking chunk of chunk_size
        chunk = tokens[:chunk_size]

        # decode the chunk
        chunked_text = tokenizer.decode(chunk)

        # skip the chunk if it is whitespace or empty
        if not chunked_text and chunked_text.isspace():
            # discard the tokens till this chunk since there is no use of whitespace or empty chunk
            tokens = tokens[len(chunk) :]

            # move to the next chunk
            continue

        # We want to divide chunks at the end of puncutations so that no semantic meaning is lost whiel breaking
        index_last_puncutation = max(
            chunked_text.rfind("."),
            chunked_text.rfind("?"),
            chunked_text.rfind("\n"),
            chunked_text.rfind("!"),
        )

        # we get -1 if there is no above mentioned punctuations in the chunked text
        # If there is punctuation and index of it is more than minimum characters present in a chunk,
        #  we need to break the chunk at the punctuation mark
        if (
            index_last_puncutation != -1
            and index_last_puncutation > MIN_CHUNK_SIZE_CHARS
        ):
            # break at the last punctuation
            chunked_text = chunked_text[: index_last_puncutation + 1]

        chunked_text_append = chunked_text.replace(
            "\n", " "
        ).strip()  # remove trailing new line characters and whitespaces

        if len(chunked_text_append) > MIN_CHUNK_LENGTH_TO_EMBED:
            chunks.append(chunked_text_append)

        # Remove the tokens till the text where we got the last punctuatation
        tokens = tokens[len(tokenizer.encode(chunked_text, disallowed_special=())) :]

        num_chunks += 1

    if tokens:
        remaining_text = tokenizer.decode(tokens).replace("\n", " ").strip()
        if len(remaining_text) > MIN_CHUNK_LENGTH_TO_EMBED:
            chunks.append(remaining_text)

    return chunks


# lets et embeddings from chunks of text
def get_embeddings(chunk: list[str], client: openai.OpenAI):
    # call the embeddings from openai itself, you can also use other embedding models from hugging face customzie this function
    embedding = (
        client.embeddings.create(input=chunk, model=EMBEDDING_MODEL).data[0].embedding
    )
    return embedding


# We need to create a mappting between the file we have and the embeddings
def get_vectorstore(file_path, client: openai.OpenAI, chunk_size=CHUNK_SIZE):
    chunks = get_text_chunk(file_path, chunk_size)
    embeddings = get_embeddings(chunks, client)
    return chunks, embeddings


def get_top_k_chunks(chunk_embeddings, query_embeddings, k=4):
    """
    The function performs similarity search between chunked embeddings and the query embeddings and
    return the top 4 indices of the chunks, which can be used by the llm to answer the query.
    """
    # initialize the similarity distance function
    csm = torch.nn.CosineSimilarity(dim=1)

    # get the distance/scores of between each chunk and query embedding
    similarity_score = csm(
        torch.tensor([chunk_embeddings]), torch.tensor(query_embeddings)
    )

    # sort them in decending order and get indices of top k indices
    top_indices = torch.sort(similarity_score, descending=True).indices[:k]
    return top_indices.tolist()


def chat_completions(client: openai.OpenAI, context: str, chat_history: list[dict]):

    # Lets define system prompt
    system_prompt = f"""\n
You are an intelligent assistant tasked to answers asked by the user from the context available to you.
1. First, you will internally assess whether the content provided is relevant to reply to the input prompt.
2. If that is the case, answer directly using this content. If the content is relevant, use elements found in the content to craft a reply to the input prompt.
3. If the content is not relevant, use your own knowledge to reply or say that you don't know how to respond if your knowledge is not sufficient to answer.
Stay concise with your answer, replying specifically to the input prompt without mentioning additional information provided in the context content.

context: \n{context}
"""

    # add user query, system prompt, along with chat history and it is where we have user input
    messages = [
        {"role": "system", "content": system_prompt},
    ] + chat_history

    # call the model with message input, set temperature 0.0 so that it will not get much creative
    response = (
        client.chat.completions.create(
            messages=messages,
            model=LLM_MODEL,
            temperature=0.0,
        )
        .choices[0]
        .message.content
    )
    return response


def main():

    # initialize the client
    client = openai.Client()
    chat_history = []
    # get the path of the file you want to ask questions on
    file_paths = get_data()

    # get the text chunks, embeddings
    chunks, chunk_embeddings = get_vectorstore(file_paths, client, chunk_size=512)

    try:
        while True:
            # get the input query from the user
            current_input = input(">>> ")

            # we add user message to chat history
            chat_history.append({"role": "user", "content": current_input})

            # get query embeddings
            query_embedding = get_embeddings([current_input], client)

            # get top k indices of chunks that are clsoe to the query
            top_k_indices = get_top_k_chunks(chunk_embeddings, query_embedding, k=4)

            # create the context string from the top - k indices we have
            context = " ".join([chunks[indice] for indice in top_k_indices])

            # get the repsosne generated by the model
            response = chat_completions(client, context, chat_history)

            # we add reponse of the model to chat history
            chat_history.append({"role": "assistant", "content": response})

            print(response)
    except KeyboardInterrupt:
        print("Session Ended")


if __name__ == "__main__":
    main()
