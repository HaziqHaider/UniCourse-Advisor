import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class RAGEngine:
    def __init__(self, pdf_directory="data/pdfs", persist_directory="vectorstore"):
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        
        # Use updated HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.retriever = None
        
    def load_and_process_documents(self):
        """Load PDFs and create vector database"""
        documents = []
        
        # Check if PDF directory exists
        if not os.path.exists(self.pdf_directory):
            os.makedirs(self.pdf_directory)
            return f"Created PDF directory at {self.pdf_directory}. Please add PDF files."
        
        # Load all PDF files
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            return "No PDF files found. Please add course catalog PDFs to the data/pdfs folder."
        
        for filename in pdf_files:
            try:
                loader = PyPDFLoader(os.path.join(self.pdf_directory, filename))
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        if not documents:
            return "No documents could be loaded from PDF files."
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        return f"Processed {len(chunks)} chunks from {len(documents)} documents across {len(pdf_files)} PDF files"
    
    def query_courses(self, question):
        """Query the course database"""
        if not self.retriever:
            raise ValueError("Please load documents first using load_and_process_documents()")
        
        relevant_docs = self.retriever.invoke(question)
        return relevant_docs
    
    def get_course_info(self, docs):
        """Extract course information from retrieved documents"""
        context = "\n\n".join([doc.page_content for doc in docs])
        return context