def filter_retrieved_documents(query, category, k=7, threshold=0.5):

    search_kwargs = {"k": k}
    if category != 'Admin':
        search_kwargs["filter"] = {'Category': category}
    docs_with_scores = vectorstore_faiss.similarity_search_with_score(query, **search_kwargs)
    sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
    filtered_docs = [doc for doc, score in sorted_docs if score >= threshold]
    return filtered_docs

class DummyRetriever:
    def __init__(self, docs):
        self.docs = docs
    def get_relevant_documents(self, _):
        return self.docs

def retrieve_docs_faiss_followup(query,embedding,is_arabic,chat_history,category):
    print("Is the question in arabic language ",is_arabic)
    print("Filter on category :",category)
    print("**********************")
    print("CHAT HISTORY",chat_history)
    if len(category) == 1:
        category = category[0]
    else:
        category = category

    
    flagging_prompt = PromptTemplate(template=flagging_prompt_template, input_variables=["question"])

    classifier_llm = LLama3LLM()  
 
    def classify_question(question):
        flagging_chain = LLMChain(llm=classifier_llm, prompt=flagging_prompt)
        classification = flagging_chain.run(question)
        return classification


    prompt = PromptTemplate(template=prompt_template2, input_variables=["context", "question"])
 
    llm = LLama3LLM()


    filtered_docs = filter_retrieved_documents(query, category, k=7, threshold=0.75)
    dummy_retriever = DummyRetriever(filtered_docs)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=dummy_retriever,
        return_source_documents=True,
        chain_type_kwargs={"verbose": True, "prompt": prompt}
    )
    
