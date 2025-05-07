# import basics
import os
import pandas as pd
import uuid
from dotenv import load_dotenv
from tqdm import tqdm
import json
import time
import psutil
import asyncio
import aiohttp
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# import langchain
from langchain_openai import OpenAIEmbeddings

# import supabase
from supabase import create_client, Client

# load environment variables
load_dotenv()

# initialize supabase db
supabase_url: str = os.getenv("SUPABASE_URL")
supabase_key: str = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initialize embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


async def get_embedding(session: aiohttp.ClientSession, text: str) -> List[float]:
    """Get embedding using OpenAI API directly with aiohttp"""
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    data = {"input": text, "model": "text-embedding-3-large"}

    async with session.post(
        "https://api.openai.com/v1/embeddings", headers=headers, json=data
    ) as response:
        result = await response.json()
        return result["data"][0]["embedding"]


async def process_batch(
    session: aiohttp.ClientSession, batch: List[Dict[str, Any]], table_name: str
) -> int:
    """Process a batch of documents concurrently"""
    tasks = []
    for doc in batch:
        tasks.append(get_embedding(session, doc["content"]))

    # Get embeddings for all documents in batch
    embeddings = await asyncio.gather(*tasks)

    # Prepare data for batch insert
    records = []
    for doc, embedding in zip(batch, embeddings):
        records.append(
            {
                "id": str(uuid.uuid4()),
                "content": doc["content"],
                "metadata": doc["metadata"],
                "embedding": embedding,
            }
        )

    # Batch insert to Supabase
    supabase.table(table_name).insert(records).execute()
    return len(records)


async def process_csv_to_db(
    file_path: str, table_name: str = "documents", batch_size: int = 10
):
    """
    Process CSV file and insert rows in batches with concurrent processing
    """
    print(f"Loading CSV file: {file_path}")

    start_memory = get_memory_usage()
    start_time = time.time()

    # Read CSV file
    df = pd.read_csv(file_path)

    # Check if required columns exist
    required_cols = ["metadata", "content"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: CSV file is missing required column: {col}")
            return

    # Prepare documents
    documents = []
    total_tokens = 0

    for _, row in df.iterrows():
        if pd.isna(row["content"]) or not row["content"]:
            continue

        content = row["content"]
        total_tokens += len(content.split())

        try:
            metadata = (
                json.loads(row["metadata"])
                if isinstance(row["metadata"], str)
                else row["metadata"]
            )
        except Exception:
            metadata = {}

        documents.append({"content": content, "metadata": metadata})

    # Process in batches
    successful_inserts = 0
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            successful_inserts += await process_batch(session, batch, table_name)
            print(
                f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents"
            )

    # Calculate performance metrics
    end_time = time.time()
    end_memory = get_memory_usage()

    print("\n=== Performance Metrics ===")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(
        f"Average processing time per document: {(end_time - start_time)/successful_inserts:.4f} seconds"
    )
    print(f"Memory usage: {end_memory - start_memory:.2f} MB")
    print(f"Total documents processed: {successful_inserts}")
    print(f"Total approximate tokens: {total_tokens}")
    print(f"Average tokens per document: {total_tokens/successful_inserts:.2f}")
    print("========================\n")

    print(f"Successfully inserted {successful_inserts} rows into {table_name}")


def main():
    # Path to CSV files
    csv_folder = "output"
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the output folder")
        return

    print(f"Found {len(csv_files)} CSV files to process")
    # Load all CSV files
    for csv_file in csv_files:
        file_path = os.path.join(csv_folder, csv_file)
        try:
            print(f"Processing file: {csv_file}")
            asyncio.run(process_csv_to_db(file_path))
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

    print("CSV documents successfully loaded into database")


if __name__ == "__main__":
    main()
