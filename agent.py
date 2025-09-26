import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool
from langgraph.graph import StateGraph, END
from typing import TypedDict
from tavily import TavilyClient
from dotenv import load_dotenv

from rag_engine import RAGEngine

load_dotenv()

class AgentState(TypedDict):
    question: str
    query_type: str
    course_context: str
    web_context: str
    final_answer: str

class IntelliCourseAgent:
    def __init__(self, rag_engine):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))
        self.rag_engine = rag_engine
        
        # Initialize Tavily only if API key exists
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            self.tavily = TavilyClient(api_key=tavily_api_key)
            self.tavily_available = True
            print("Tavily web search enabled successfully")
        else:
            self.tavily_available = False
            print("Tavily API key not found. Web search will be disabled.")
        
        # Build the graph
        self.graph = self._build_graph()
    
    def retrieve_course_info(self, query):
        """Retrieve course information from vector database"""
        docs = self.rag_engine.query_courses(query)
        return self.rag_engine.get_course_info(docs)
    
    def search_web(self, query):
        """Search the web for general information"""
        if not self.tavily_available:
            return "Web search is currently unavailable. Please check your Tavily API key."
        
        try:
            response = self.tavily.search(query=query, max_results=3)
            return "\n".join([result["content"] for result in response["results"]])
        except Exception as e:
            return f"Web search error: {str(e)}"
    
    def router(self, state: AgentState) -> dict:  # CHANGED: Return dict instead of str
        """Determine if query is course-related or general"""
        question = state["question"]
        
        router_prompt = f"""
        Classify the following user question as either "course" or "general":
        
        Question: {question}
        
        Return only one word: "course" or "general"
        
        Course questions are about: specific courses, prerequisites, curriculum, majors, departments, university programs.
        General questions are about: career advice, job market, general concepts, non-university topics.
        """
        
        response = self.llm.invoke(router_prompt)
        query_type = response.content.strip().lower()
        
        # Return a dictionary that updates the state
        return {"query_type": query_type}
    
    def course_retrieval_node(self, state: AgentState) -> dict:
        """Handle course-related queries"""
        context = self.retrieve_course_info(state["question"])
        return {"course_context": context}
    
    def web_search_node(self, state: AgentState) -> dict:
        """Handle general knowledge queries"""
        context = self.search_web(state["question"])
        return {"web_context": context}
    
    def generate_answer_node(self, state: AgentState) -> dict:
        """Generate final answer based on retrieved context"""
        question = state["question"]
        course_context = state.get("course_context", "")
        web_context = state.get("web_context", "")
        
        # Combine contexts
        if course_context and web_context:
            context = f"Course Information:\n{course_context}\n\nGeneral Information:\n{web_context}"
        else:
            context = course_context + web_context
        
        prompt = f"""
        You are IntelliCourse, a helpful university course advisor. Answer the student's question based on the context provided.
        
        Context: {context}
        
        Question: {question}
        
        Provide a clear, helpful answer. If the context doesn't contain enough information, say so politely.
        """
        
        response = self.llm.invoke(prompt)
        return {"final_answer": response.content}
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.router)
        workflow.add_node("course_retrieval", self.course_retrieval_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # FIXED: Correct conditional edges logic
        def route_condition(state: AgentState):
            query_type = state.get("query_type", "").lower()
            if query_type == "course":
                return "course_retrieval"
            else:
                return "web_search"
        
        workflow.add_conditional_edges(
            "router",
            route_condition,  # Use the function, not lambda
            {
                "course_retrieval": "course_retrieval",
                "web_search": "web_search"
            }
        )
        
        # Add edges to final answer generation
        workflow.add_edge("course_retrieval", "generate_answer")
        workflow.add_edge("web_search", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def query(self, question: str) -> str:
        """Main method to query the agent"""
        initial_state = AgentState(
            question=question, 
            query_type="", 
            course_context="", 
            web_context="", 
            final_answer=""
        )
        result = self.graph.invoke(initial_state)
        return result["final_answer"]