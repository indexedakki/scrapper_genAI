# LinkScrapper

LinkScrapper is an AI-powered web application that helps you generate insightful questions about any webpage. It combines the power of Mixtral models from Hugging Face with Pinecone's vector database for efficient storage and retrieval of word embeddings.

## Features

- User authentication system
- Webpage analysis using Mixtral models
- Question generation based on webpage content
- Integration with Pinecone vector database for fast embeddings lookup
- Clean and intuitive user interface built with React

## Technologies Used

- Frontend: React.js
- Backend: FastAPI
- AI Model: Mixtral (Hugging Face)
- Vector Database: Pinecone
- Authentication: JWT-based

## Getting Started

### Prerequisites

- Node.js (version 14 or higher)
- Python (version 3.7 or higher)

### Installation
1. Clone this repository:
```bash
git clone https://github.com/indexedakki/scrapper_genAI.git
git clone https://github.com/indexedakki/scrapper_UI.git
```
2. Install dependancies:
```bash
cd scrapper_genAI
pip install -r requirements.txt
cd scrapper_UI
npm install
```
3. Run React and fastapi
```bash
uvicorn main:app --reload
npm start
```
## Usage

1. Sign up for an account
2. Log in to your account
3. Paste a URL in the input field
4. The page source will be scraped in the background
5. Ask specific questions relevant to the webpage content

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
