import streamlit as st
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQA
from llm import llm, embeddings
from langchain.chains import RetrievalQA

neo4jvector = Neo4jVector.from_existing_index(
    embedding=embeddings,                        
    url=st.secrets["NEO4J_URI"],             
    username=st.secrets["NEO4J_USERNAME"],   
    password=st.secrets["NEO4J_PASSWORD"],   
    index_name="tweet_texts"              
#     node_label="Tweet",                      
#     text_node_property="text",               
#     embedding_node_property="text_embedding",
#     retrieval_query="""
# RETURN
# node.text AS text,
# {
#     author: [ (user)-[:POSTS]->(node) | user.name ]
# } AS metadata
# """
)

retriever = neo4jvector.as_retriever()
kg_qa = RetrievalQA.from_chain_type(
    llm,                  # <1>
    chain_type="stuff",   # <2>
    retriever=retriever,  # <3>
)