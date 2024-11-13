from itertools import chain
import os
import tempfile
import ast
import uuid
import logging
import traceback
from datetime import datetime
from typing import List, Optional


from fastapi import APIRouter, Form, HTTPException, Body, File, UploadFile
from fastapi.responses import JSONResponse

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import PostgresChatMessageHistory
import numpy as np

from database import supabase
from schemas import Message, chat_schema, Session
from utils import simulate_ai_processing_time

# Initialize logging
logging.basicConfig(level=logging.INFO)

router: APIRouter = APIRouter()

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Get the Google API key
google_api_key: str = os.getenv("GOOGLE_API_KEY")

# Initialize the language model with a valid API key
llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash", google_api_key=google_api_key
)

# Define the prompt template
prompt: PromptTemplate = PromptTemplate(
    input_variables=["human_message"], template="Human: {human_message}\nAI:"
)

# Setup PostgresChatMessageHistory
connection_string: str = os.getenv("POSTGRES_CONNECTION_STRING")


@router.get("/")
async def read_root() -> dict:
    """Root endpoint that provides a welcome message."""
    return {
        "message": "Welcome to the Chat API. Use the /chat endpoint to interact with the chatbot."
    }


@router.post("/chat")
async def chat(request: chat_schema) -> dict:
    """Handles chat requests and generates responses from the language model."""
    try:
        logging.info(f"Received message: {request.text}")

        session_id: str = request.session_id
        logging.info(f"Using session ID: {session_id}")

        # Initialize PostgresChatMessageHistory
        chat_history: PostgresChatMessageHistory = PostgresChatMessageHistory(
            connection_string=connection_string, session_id=session_id
        )

        # Initialize ConversationSummaryMemory with the LLM
        memory: ConversationSummaryMemory = get_summary_memory(
            llm=llm, chat_memory=chat_history
        )

        # Add the user message to memory
        memory.chat_memory.add_user_message(request.text)

        # Determine if the PDFs are relevant to the question
        is_related_to_pdfs = await should_query_pdfs(request.text, session_id)

        if is_related_to_pdfs:
            # Directly call the query_pdf function to get relevant content
            pdf_responses = await internal_query_pdf(request.text, session_id)
            context = "\n\n".join(pdf_responses)
            prompt_template = get_pdf_aware_prompt_template()
            vector_store_used = True
        else:
            context = ""  # No relevant PDF content
            prompt_template = prompt  # Use the general prompt
            vector_store_used = False

        # Generate a response using the appropriate prompt template
        chain = get_llm_chain(llm, memory, prompt_template)

        ai_response = chain.run(
            {
                "human_message": request.text,
                "context": context,
                "vector_store_used": vector_store_used,
            }
        )

        logging.info(f"Model response: {ai_response}")

        # Add the AI's response to memory
        memory.chat_memory.add_ai_message(ai_response)

        # Store chat history in Supabase
        response = (
            supabase.table("chat_history")
            .insert(
                [
                    {
                        "session_id": session_id,
                        "message": request.text,
                        "role": "human",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    {
                        "session_id": session_id,
                        "message": ai_response,
                        "role": "ai",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                ]
            )
            .execute()
        )

        logging.info(f"Supabase response: {response}")

        if not response.data:
            logging.error(f"Failed to store chat history: {response}")
            raise HTTPException(status_code=500, detail="Failed to store chat history.")

        # Update session last updated timestamp
        update_response = (
            supabase.table("sessions")
            .update({"last_updated": datetime.utcnow().isoformat()})
            .eq("session_id", session_id)
            .execute()
        )
        logging.info(f"Session update response: {update_response}")

        return {"reply": ai_response, "vector_store_used": vector_store_used}

    except Exception as e:
        traceback.print_exc()
        logging.critical(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    # Define the prompt template for when PDF content is relevant


def get_summary_memory(
    llm: ChatGoogleGenerativeAI,
    chat_memory: PostgresChatMessageHistory,
    memory_key: str = "history",
    input_key: str = "human_message",
) -> ConversationSummaryMemory:
    """Returns a ConversationSummaryMemory instance."""
    return ConversationSummaryMemory(
        llm=llm, memory_key=memory_key, input_key=input_key, chat_memory=chat_memory
    )


def get_llm_chain(
    llm: ChatGoogleGenerativeAI,
    memory: ConversationSummaryMemory,
    prompt: PromptTemplate,
    verbose: bool = False,
) -> LLMChain:
    """Returns an LLMChain instance."""
    return LLMChain(llm=llm, memory=memory, verbose=verbose, prompt=prompt)


async def should_query_pdfs(message: str, session_id: str) -> bool:
    pdf_response = await internal_query_pdf(message, session_id)
    logging.info(f"PDF query response: {pdf_response}")

    return (
        bool(pdf_response)
        and pdf_response[0] != "No relevant information found in the documents."
    )


def get_pdf_aware_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        template=(
            "You are an assistant, answer the following query.\n"
            "Context from documents: ## {context} ##\n"
            "Query: {human_message}\n"
            "Vector Store Used: {vector_store_used}\n"
            "If relevant information is in the context, include it in your response. "
            "Avoid mentioning that you are using an AI or PDF documents.\n"
        ),
    )



async def internal_query_pdf(query: str, session_id: str) -> List[str]:
    try:
        # Fetch session data
        session_response = (
            supabase.table("sessions")
            .select("document_ids")
            .eq("session_id", session_id)
            .execute()
        )

        if not session_response.data:
            raise HTTPException(status_code=404, detail="Session not found.")

        document_ids = session_response.data[0]["document_ids"]

        if not document_ids:
            raise HTTPException(
                status_code=404, detail="No documents found for the session."
            )

        # Fetch document data
        documents_response = (
            supabase.table("documents")
            .select("content", "embedding")
            .in_("id", document_ids)
            .execute()
        )

        if not documents_response.data:
            raise HTTPException(status_code=404, detail="No documents found.")

        # Generate query embedding
        query_embedding = np.array(get_embedding(query), dtype=np.float32)

        relevant_responses = []

        for document in documents_response.data:
            document_embedding_str = document["embedding"]
            
            # Log the document content and a message indicating the presence of embeddings
            logging.info(f"Processing document: {document['content'][:50]}... (embedding available)")

            # Check if the document embedding is already a list or string
            if isinstance(document_embedding_str, str):
                # If it's a string, try to parse it into a list
                if not (document_embedding_str.startswith("[") and document_embedding_str.endswith("]")):
                    logging.error(f"Invalid embedding format for document: {document['content']}")
                    continue  # Skip this document if the embedding format is invalid
                try:
                    document_embedding = np.array(
                        ast.literal_eval(document_embedding_str), dtype=np.float32
                    )
                except (ValueError, SyntaxError) as e:
                    logging.error(f"Error parsing embedding for document {document['content']}: {e}")
                    continue  # Skip this document if embedding cannot be parsed
            elif isinstance(document_embedding_str, list):
                # If the embedding is already a list, convert it directly to np.array
                document_embedding = np.array(document_embedding_str, dtype=np.float32)
            else:
                logging.error(f"Unexpected embedding format for document: {document['content']}")
                continue  # Skip this document if the embedding format is neither string nor list

            # Compute similarity
            similarity = np.dot(query_embedding, document_embedding)
            logging.info(f"Similarity score computed for document: {document['content'][:50]}...")

            # Add to responses if similarity is above threshold
            if similarity > 0.5:  # Adjust as needed
                relevant_responses.append(document["content"])

        if not relevant_responses:
            return []

        return relevant_responses

    except Exception as e:
        logging.error(f"Error querying PDFs: {str(e)}")
        return []
    
@router.post("/add_pdf/{session_id}")
async def add_pdf(session_id: str, files: List[UploadFile] = File(...)) -> dict:
    """Handles multiple PDF file uploads and associates the files with a session."""
    try:
        document_ids = []

        for file in files:
            # Create a temporary file to store the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(await file.read())
                temp_file_path = temp_file.name

            # Load the PDF and extract text content
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load_and_split()
            document_content = "\n".join([page.page_content for page in pages])

            # Generate a UUID for the document
            document_id = str(uuid.uuid4())

            # Store the document metadata and content in Supabase
            response = (
                supabase.table("documents")
                .insert(
                    {
                        "id": document_id,
                        "session_id": session_id,
                        "content": document_content,
                        "metadata": {"filename": file.filename},
                        "embedding": get_embedding(document_content),
                    }
                )
                .execute()
            )

            if not response.data:
                raise HTTPException(status_code=500, detail="Failed to store document.")

            # Append the new document ID to the list
            document_ids.append(document_id)

        # Fetch the existing document_ids for the session
        session_response = (
            supabase.table("sessions")
            .select("document_ids")
            .eq("session_id", session_id)
            .execute()
        )
        existing_document_ids = session_response.data[0]["document_ids"]

        # Append the new document IDs to the existing document_ids array
        updated_document_ids = existing_document_ids + document_ids

        # Update the session with the new document_ids array
        update_response = (
            supabase.table("sessions")
            .update(
                {
                    "document_ids": updated_document_ids,
                    "last_updated": datetime.utcnow().isoformat(),
                }
            )
            .eq("session_id", session_id)
            .execute()
        )

        return {"message": "PDFs uploaded and processed successfully"}
    except Exception as e:
        error_message = {"detail": f"An error occurred: {str(e)}"}
        logging.error(error_message)
        return JSONResponse(content=error_message, status_code=500)


def get_embedding(document_content: str) -> List[float]:
    """Generate embeddings using Google Generative AI Embeddings."""
    # Initialize embeddings model
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Generate embeddings for the document content using the correct method
    embedding = embeddings_model.embed_documents([document_content])[0]

    return embedding


@router.get("/history/{session_id}")
async def get_history(session_id: str) -> list:
    """Fetches the chat history for a given session ID."""
    try:
        logging.info(f"Fetching history for session_id: {session_id}")
        response = (
            supabase.table("chat_history")
            .select("message, role, timestamp")
            .eq("session_id", session_id)
            .order("timestamp")
            .execute()
        )
        return response.data if response.data else []
    except Exception as e:
        logging.critical(f"Error fetching history: {str(e)}")
        return []

@router.post("/session", response_model=Session)
async def create_session(email_id: str = Body(..., embed=True)) -> JSONResponse:
    """Creates a new chat session."""
    try:
        session_id: str = str(uuid.uuid4())
        timestamp: str = datetime.utcnow().isoformat()

        response = (
            supabase.table("sessions")
            .insert(
                {
                    "session_id": session_id,
                    "started_at": timestamp,
                    "last_updated": timestamp,
                    "email_id": email_id,
                    "document_ids": [],  # Initialize with an empty list
                }
            )
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to create session.")

        new_session: dict = {
            "session_id": session_id,
            "email_id": email_id,
            "started_at": timestamp,
        }

        return JSONResponse(content=new_session, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{email_id}", response_model=List[Session])
async def get_sessions(email_id: str) -> JSONResponse:
    """Fetches all sessions for a given email ID, with document summary if available."""
    try:
        response = (
            supabase.table("sessions")
            .select("session_id, started_at, document_ids")
            .eq("email_id", email_id)
            .execute()
        )
        sessions = response.data

        for session in sessions:
            if session.get("document_ids"):
                # Fetch document metadata to display as the session name
                doc_ids = session["document_ids"]
                descriptions = []
                for doc_id in doc_ids:
                    doc_response = (
                        supabase.table("documents")
                        .select("metadata")
                        .eq("id", doc_id)
                        .execute()
                    )
                    if doc_response.data:
                        metadata = doc_response.data[0]["metadata"]
                        descriptions.append(metadata.get("filename", "Unnamed PDF"))
                session["pdf_descriptions"] = ", ".join(descriptions)

        if not sessions:
            raise HTTPException(status_code=404, detail="No sessions found.")

        return JSONResponse(content=sessions, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def query_pdf(session_id: str, query: str) -> dict:
    """Extracts information from the PDF content stored as vectors."""
    try:
        # Fetch the document_ids for the session
        session_response = (
            supabase.table("sessions")
            .select("document_ids")
            .eq("session_id", session_id)
            .execute()
        )

        if not session_response.data:
            raise HTTPException(status_code=404, detail="Session not found.")

        document_ids = session_response.data[0]["document_ids"]

        if not document_ids:
            raise HTTPException(
                status_code=404, detail="No documents found for the session."
            )

        # Fetch the content and embeddings for all documents in the session
        documents_response = (
            supabase.table("documents")
            .select("content", "embedding")
            .in_("id", document_ids)
            .execute()
        )

        if not documents_response.data:
            raise HTTPException(status_code=404, detail="No documents found.")

        # Extract the embeddings for the query
        query_embedding = get_embedding(query)

        # Find the most relevant document by comparing the embeddings
        most_relevant_content = None
        highest_similarity = -1

        for document in documents_response.data:
            document_embedding = document["embedding"]

            # Compute the similarity (using cosine similarity or dot product)
            similarity = np.dot(query_embedding, document_embedding)

            if similarity > highest_similarity:
                highest_similarity = similarity
                most_relevant_content = document["content"]

        if most_relevant_content is None:
            raise HTTPException(status_code=404, detail="No relevant content found.")

        return {"response": most_relevant_content}

    except Exception as e:
        error_message = {"detail": f"An error occurred: {str(e)}"}
        logging.error(error_message)
        return JSONResponse(content=error_message, status_code=500)