# utils.py
import os, time, io, csv
from pydantic import Field
import requests
import boto3
from datetime import datetime
from typing import Union
from app.api.api_v1.models import Datasource
from pprint import pprint

# App
from app.api.api_v1.database import upsert_datasource

# LLM and Tool components
from openai import OpenAI
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# diff doc loaders
from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader

# scraping
from bs4 import BeautifulSoup as Soup

""" Locations """
temp_downloads_path = "/app/app/temp_downloads"

bucket_downloads_path = "api_ingress"


def get_pinecone_index(
    index_name: str = os.environ["PINECONE_INDEX_NAME"],
    dimension: int = 1536,
    metric: str = "cosine",
):
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENV"],
    )
    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(index_name, dimension=dimension, metric=metric)
        # wait for index to be initialized
        while not pinecone.describe_index(index_name).status["ready"]:
            time.sleep(1)
    return pinecone.Index(index_name)


def download_file(url: str, local_filename: str | None = None) -> Union[str, bool]:
    try:
        if local_filename is None:
            local_filename = f"{temp_downloads_path}/{url.rsplit('/', 1)[1]}"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return local_filename
    except requests.RequestException as e:
        print(f"Error: {e}")
        return False


#
def download_file_to_s3(
    url: str,
    bucket: str = os.environ["RAG_BUCKET"],
    filename: str | None = None,
    add_file_ext: str | None = None,
) -> Union[str, bool]:
    """Downloads a file from a url to the specified bucket and filename (full path).  On success returns the
    full s3 link"""
    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3 = session.resource("s3")
    buff = io.BytesIO()

    if filename is None:
        filename = f"{bucket_downloads_path}/{url.rsplit('/', 1)[1]}"
        if add_file_ext is not None:
            filename += add_file_ext
        # print(f">>>>>>>>>>>>USING THIS FILENAME {filename}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                buff.write(chunk)
        s3.Object(bucket, filename).put(Body=buff.getvalue())
        return f"s3://{bucket}/{filename}"
    except requests.RequestException as e:
        print(f"Error: {e}")
        return False


# metadata_to_save is a list of dicts with key value pairs of desired metadata to save.
# need to update to use https://python.langchain.com/docs/integrations/document_loaders/recursive_url
# for websites
async def load_file(
    file_type: str,
    file_link: str,
    metadata_to_save: list[dict],
    parser_args: dict | None = None,
    recursive_scraping: bool | None = True,
) -> Union[str, bool]:
    """Loads file into Pinecone and/or Mongo as well as saving the original file to S3"""
    # Always put some basics in the meta for more restrictive searches
    metadata_to_save.append({"indexed_datetime_utc": datetime.utcnow()})
    metadata_to_save.append({"source": file_link})
    # keep track of what the file started as
    metadata_to_save.append({"original_file_type": file_type})

    # Helps keep track of if any cleanup needs to happen
    local_file_link = False
    # get rid of trailing slash, messes up naming.
    file_link = file_link.rstrip("/")
    try:
        # choose the loader based on the file_type
        match file_type.lower():
            case "pdf":
                # https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.pdf.OnlinePDFLoader.html
                loader = OnlinePDFLoader(file_link)
            # CSV's shouldnt be vectorized more than likely and should just be
            # saved and loaded into memory.  The question becomes how to know they are there (for the LLM) and when to use them (metadata?)
            # then just use the document loader and an analysis tool like pandas and/or seaborn.
            case "csv":
                # https://python.langchain.com/docs/integrations/document_loaders/csv
                # optional csv parser args that can be put in parser_args
                # csv_args = (
                #     {
                #         "delimiter": ",",
                #         "quotechar": '"',
                #         "fieldnames": ["MLB Team", "Payroll in millions", "Wins"],
                #     },
                # )

                local_file_link = download_file(url=file_link)
                # print(local_file_link)

                # Broken
                # with open(local_file_link) as fp:
                #     reader = csv.reader(fp)
                #     headers = next(reader)  # The header row is now consumed
                #     ncol = len(headers)
                #     nrow = sum(1 for _ in reader)  # What remains are the data rows
                # #
                # pprint(headers)

                # Don't create embeddings for Csvs, so just save and return here.

                s3_key = download_file_to_s3(url=file_link)
                os.unlink(local_file_link)
                return s3_key

            case "doc" | "docx":
                # https://python.langchain.com/docs/integrations/document_loaders/microsoft_word
                loader = Docx2txtLoader(file_link)
            case "website" | "html":
                if recursive_scraping:
                    # loader = RecursiveUrlLoader(url=file_link)
                    print(
                        "_________________________DOING A RECURSIVE SCRAPE___________________________"
                    )
                    loader = RecursiveUrlLoader(
                        url=file_link,
                        max_depth=5,
                        extractor=lambda x: Soup(x, "html.parser").text,
                    )

                else:
                    loader = AsyncHtmlLoader([file_link])

            case _:  # default
                raise ValueError(f"The file_type {file_type} is not yet suported bro!")

        data = loader.load()

        # Transformers if needed:
        # Probably need to use a different splitter too for things like Markdown
        # see this article:  https://www.pinecone.io/learn/chunking-strategies/
        match file_type.lower():
            case "html" | "website":
                html2text = Html2TextTransformer()
                data = html2text.transform_documents(data)

        # Continue
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=48,
            length_function=len,
            add_start_index=True,
        )

        texts = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENV"],
        )

        index_name = pinecone.Index(os.environ["PINECONE_INDEX_NAME"])

        # load the metadata into each record to be inserted.
        i = 0
        for t in texts:
            for mts in metadata_to_save:
                key, value = list(mts.items())[0]
                texts[i].metadata[key] = value
            i = i + 1
        # load the data to the vs
        v = Pinecone.from_texts(
            texts=[t.page_content for t in texts],
            embedding=embeddings,
            index_name=os.environ["PINECONE_INDEX_NAME"],
            # namespace=os.environ["PINECONE_NAMESPACE"],
            metadatas=[t.metadata for t in texts],
        )

        # Save for posterity to S3
        # the transformer turns this into markdown, maybe use .md
        if file_type == "website" or file_type == "html":
            add_file_ext = ".html"
        else:
            add_file_ext = None

        s3_key = download_file_to_s3(url=file_link, add_file_ext=add_file_ext)

        # Cleanup
        if local_file_link:
            os.unlink(local_file_link)

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise Exception(err)

    return s3_key


def pinecone_doc_exists(doc_link: str) -> bool:
    """Look at metadata source field to see if document already in the vsdb"""
    print(f"searching for source {doc_link}")
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    index = pinecone.Index(os.environ["PINECONE_INDEX_NAME"])
    # syntax only allows for exact searches, not mongo style evn tho looks sim.
    q_res = index.query(
        vector=[0] * 1536,  # required to put vector
        top_k=1,
        # namespace=os.environ["PINECONE_NAMESPACE"],
        include_metadata=True,
        filter={"source": {"$eq": doc_link}},
    )
    if len(q_res["matches"]) > 0:
        return True
    else:
        print("no matches found")
        return False


async def search_pinecone(
    query: str,
    embed_model_id: str = "text-embedding-ada-002",
    top_k: int = 3,
    texts_only: bool = False,
    metadata_filter: dict | None = None,
):
    # create query embedding

    client = OpenAI()

    print(f"Searching for {query} ")

    xq = client.embeddings.create(input=[query], model=embed_model_id).data[0].embedding

    if metadata_filter is not None:
        metadata_filter = {}

    res = get_pinecone_index().query(
        xq,
        top_k=top_k,
        include_metadata=True,
        # namespace=os.environ["PINECONE_NAMESPACE"],
        filter=metadata_filter,
    )
    # if texts_only:
    #     contexts = [x["metadata"]["text"] for x in res["matches"]]
    # else:
    #     # pprint([m  for m in res['matches']])
    #     # get list of retrieved texts
    #     # contexts = [x['metadata']['text'] for x in res['matches']]
    #     contexts = [x for x in res["matches"]]

    # .....need to run compression..... #
    contexts = [x for x in res["matches"]]

    formatted_contexts = f"\n{'-' * 100}\n".join(
        [
            f"Document {i+1} Score {d['score']} Source {d['metadata']['source']}:\n\n"
            + d["metadata"]["text"]
            for i, d in enumerate(contexts)
        ]
    )
    return formatted_contexts
