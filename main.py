from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re
import logging
from sqlalchemy.orm import Session
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
from langchain.text_splitter import CharacterTextSplitter
import transformers
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import requests
import os
from huggingface_hub import InferenceClient
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
import secrets
from typing import Annotated
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import hashlib
from fastapi.security import OAuth2PasswordBearer
from typing import Optional
from datetime import datetime, timedelta
from passlib.context import CryptContext

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(api_key=HUGGINGFACEHUB_API_TOKEN)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# os.environ["HUGGING_FACE_HUB_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
# login(token=HUGGINGFACEHUB_API_TOKEN)

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=PINECONE_API_KEY)

security = HTTPBasic()

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(255))

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    return hashlib.sha256(provided_password.encode()).hexdigest() == stored_password

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class checkUsers(BaseModel):
    email: str
    password: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/checkUsers")
def get_current_username(request: checkUsers, db: SessionLocal = Depends(get_db)):
    user = db.query(User).filter(User.username == request.email).first()
    if not user or not verify_password(user.hashed_password, request.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    # access_token_expires = timedelta(minutes=60)
    # access_token = create_access_token(
    #     data={"sub": request.email, "expires_delta":access_token_expires}
    # )
    
    # return {"access_token": access_token, "token_type": "bearer"}
    return {"User Authenticated successfully"}

@app.post("/logout")
async def logout_user(token: str = Depends(oauth2_scheme), db: SessionLocal = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
               
        # Here you would typically invalidate the token on the server-side
        # This could involve updating a database flag or deleting the token 
        # For this example, we'll just print a message
        print(f"User {username} logged out successfully")
        
        return {"message": "Logged out successfully"}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/users/me")
def read_current_user(username: Annotated[str, Depends(get_current_username)]):
    return {"username": username}

class Users(BaseModel):
    email: str
    password: str
    confirmPassword: str

@app.post("/users")
def create_user(request: Users, db: SessionLocal = Depends(get_db)):
    hashed_password = hash_password(request.password)
    new_user = User(username=request.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"username": new_user.username}

# Initialize the database before running the application
Base.metadata.create_all(bind=engine)

class ScrapeRequest(BaseModel):
    url: str

class Question(BaseModel):
    question: str

class ScrapeResponse(BaseModel):
    headings: list[str] = []
    paragraphs: list[str] = []
    text_summary: str = ""
    
def extract_important_tags(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove all script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    # Extract important tags
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    paragraphs = soup.find_all('p')
    
    return {
        'headings': [h.text.strip() for h in headings],
        'paragraphs': [p.text.strip() for p in paragraphs],
        'text_summary': text[:500] + '...' if len(text) > 500 else text
    }


@app.post("/scrape")
async def scrape_website(request: ScrapeRequest):
    try:
        driver = webdriver.Chrome()
        
        driver.get(request.url)

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        page_source = driver.page_source
        
        driver.quit()

        extracted_data = extract_important_tags(page_source)
        
        data = ScrapeResponse(
            headings=extracted_data['headings'],
            paragraphs=extracted_data['paragraphs'],
            text_summary=extracted_data['text_summary']
        )
        data_dict = data.dict()

        # Removing previous vectors for newer article
        try:
            index = pc.Index("website-scrapper")
            index.delete(delete_all=True, namespace='example-namespace')
        except Exception as e:
            pass

        # data_dict = app.state.data_dict
        large_paragraph = str(data_dict)

        text_splitter = CharacterTextSplitter(
            separator = ".",
            chunk_size= 200,
            chunk_overlap = 20,
            length_function = len
        )
        data_split = text_splitter.split_text(large_paragraph)

        output_data = []

        for idx, item in enumerate(data_split, start=1):
            output_data.append({
                "id": f"vec{idx}",
                "text": item
            })

        # print(output_data)
        # Convert the text into numerical vectors that Pinecone can index
        embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[d['text'] for d in output_data],
            parameters={"input_type": "passage", "truncate": "END"}
        )
        # print(embeddings)

        # Target the index where you'll store the vector embeddings
        index = pc.Index("website-scrapper")
        # Prepare the records for upsert
        # Each contains an 'id', the embedding 'values', and the original text as 'metadata'
        records = []
        for d, e in zip(output_data, embeddings):
            records.append({
                "id": d['id'],
                "values": e['values'],
                "metadata": {'text': d['text']}
            })

        # Upsert the records into the index
        index.upsert(
            vectors=records,
            namespace="example-namespace"
        )

        return {"status": "Scraping completed", "data_dict": data_dict}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/answer")
async def answer_question(request: Question):
    # # try:
    # Define your query
    query = request.question

    # Convert the query into a numerical vector that Pinecone can search with
    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )
    index = pc.Index("website-scrapper")
    # Search the index for the three most similar vectors
    results = index.query(
        namespace="example-namespace",
        vector=query_embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    # print(results)
    # print(type(results))

    # Convert the QueryResponse to a dictionary
    dict_response = results.to_dict()

    # print(dict_response)
    # print(type(dict_response))

    metadata_values = [match['metadata']['text'] for match in dict_response['matches']]
    # print(metadata_values)

    messages = [
        {
        "role": "user",
        "content": f"""{query} Answer based on context, you are given with headings, paragraphs and text summary from a website HTML 
        (You can use little help from internet to answer): context: {metadata_values}"""
        }
    ]

    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3", 
        messages=messages, 
        max_tokens=500
    )

    response = completion.choices[0].message.content

    return {"response": response}
    
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))



