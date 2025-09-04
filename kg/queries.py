from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import GraphCypherQAChain
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY not set. Add it to your .env file.")

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="abcdefgh"
)

graph.refresh_schema()  # LLM needs schema info

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    temperature=0,
    convert_system_message_to_human=True,
)

chain = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=True,allow_dangerous_requests=True)

# Example natural language query
while True:
    a = input()
    if(a == "exit"):
        break
    response = chain.run(a)
    print(response)
