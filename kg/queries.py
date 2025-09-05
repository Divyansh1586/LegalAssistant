# graph_qa_neo4j_gemini.py
import os
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate # Import PromptTemplate

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY not set. Add it to your .env file.")

# 1. Initialize the Neo4j Graph connection
try:
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="abcdefgh"  # Replace with your actual password
    )
    graph.refresh_schema()
except Exception as e:
    print(f"Failed to connect to Neo4j Graph. Error: {e}")
    exit()

# 2. Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

# 3. Create a custom prompt template with detailed instructions for the LLM
CYPHER_PROMPT = """
You are a Neo4j expert. Given the following graph schema, write a precise and efficient Cypher query to answer the user's question.

The graph contains the following entities and relationships:
* Node labels: Article, Part, Schedule, Amendment, CaseLaw, Concept, Subject
* Relationship types: CONTAINS, REFERS_TO, AMENDED_BY, INTERPRETS, MENTIONS, HAS_SUBJECT

The properties on nodes and relationships are:
* Nodes with label 'Article' have a `number` property (e.g., 'Article:21A').
* Nodes with label 'Part' have a `number` property (e.g., 'Part:II').
* Nodes with label 'Schedule' have a `number` property (e.g., 'Schedule:I').
* Relationships have a `snippet` property that provides context for the relationship.

Instructions:
1.  **Use the correct properties:** For Articles, Parts, and Schedules, use the `number` property for filtering (e.g., `WHERE a.number = '21A'`).
2.  **Do not use the `id` property** of the node.
3.  **Return only the Cypher query**, without any prose, explanations, or backticks.
4.  **Avoid generating incorrect queries** such as those that use inexact matching.

Graph Schema:
{schema}

Question:
{question}
"""
prompt = PromptTemplate.from_template(CYPHER_PROMPT)

# 4. Create the GraphCypherQAChain with the custom prompt
chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=prompt # Pass the custom prompt here
)

# 5. Interactive loop for natural language queries
print("Neo4j Graph QA Chain is ready. Type your questions about the graph or 'exit' to quit.")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    
    try:
        response = chain.invoke({"query": query})
        print("AI:", response["result"])
    except Exception as e:
        print(f"An error occurred: {e}")