from langchain_core.retrievers import BaseRetriever
from langchain.schema.document import Document
from pydantic import Field

class CustomRetriever(BaseRetriever):
    docs: List[Document] = Field(default_factory=list)
    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs
        
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self.docs

def filter_retrieved_documents(query, category, k=15, threshold=0.7):
    """Get documents with their similarity scores and maintain original retrieval order."""
    search_kwargs = {"k": k}
    if category != 'Admin':
        search_kwargs["filter"] = {'Category': category}
    
    docs_with_scores = vectorstore_faiss.similarity_search_with_score(query, **search_kwargs)
    
    sorted_docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
    
    filtered_docs = []
    scores = []
    for doc, score in sorted_docs_with_scores:
        doc.metadata['similarity_score'] = float(score)
        doc.metadata['retrieval_rank'] = len(filtered_docs)
        scores.append(float(score))
    
    return filtered_docs, scores

def retrieve_docs_faiss_followup(query, embedding, is_arabic, chat_history, category):
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


    filtered_docs, scores = filter_retrieved_documents(query, category, k=15, threshold=0.75)
    retriever = CustomRetriever(docs=filtered_docs)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"verbose": True, "prompt": prompt}
    )
    
            
    def answer_question(question,chat_history):
        document_name = []
        page_number = []
        result = {}
        if is_arabic:
            translated_question = translate_arabic_question(question)
            print("****************************************")
            print('TRANSLATED QUESTION',translated_question)
            print("****************************************")

            if(classify_question(translated_question)=="Yes"):
                result['result'] = "Hi I am a QnA bot, please feel free to ask a relevant question."
                results={'response':result['result'],'documents':[],'page_numbers':[]}
            else:
                result = qa({"query": translated_question})
                print(f"DEBUG - Query: {translated_question}")
                print(f"DEBUG - Result: {result}")
                if(("I don't know, please ask another question" in result['result']) or ("I apologize" in result['result'])):
                    print("inside elif")
                    results={'response':result['result'],'documents':[],'page_numbers':[]}
                else:
                    print("inside else")
                    if len(result['source_documents']) == 0:
                        result['result'] = "No information available in the LoB documents."
                        document_name = []
                        page_number = []
                    else:
                        for document in result['source_documents']:
                            file = document.metadata['file_path'].split('/')[-1]
                            page = document.metadata['page']+1
                            document_name.append(file)
                            page_number.append(page)
        
                    results={'response':result['result'],'documents':document_name,'page_numbers':page_number}
                    
            print('RESULT FROM RETRIVAL QA',result)
            final_answer = refine_arabic_answer(results['response'])
            print('TRANSLATED ARABIC ANSWER',final_answer)
            results['response'] = final_answer
            return results
        else:
            if(classify_question(question)=="Yes"):
                result['result'] = "Hi I am a QnA bot, please feel free to ask a relevant question."
                results={'response':result['result'],'documents':[],'page_numbers':[]}
            else:

                contexualized_question = followup_question([chat_history,question])
                print("-----------------------------------")
                print("this is contexual question")
                print(contexualized_question)
                print("-----------------------------------")
                result = qa({"query": contexualized_question})
                print(f"DEBUG - Query: {contexualized_question}")
                print(f"DEBUG - Result: {result}")
                if(("I don't know, please ask another question" in result['result']) or ("I apologize" in result['result'])):
                    print("inside elif")
                    results={'response':result['result'],'documents':[],'page_numbers':[]}
                else:
                    print("inside else")
                    if len(result['source_documents']) == 0:
                        result['result'] = "No information available in the LoB documents."
                        document_name = []
                        page_number = []
                    else:
                        ranked_docs = []
                        
                        for document in result['source_documents']:
                            file = document.metadata['file_path'].split('/')[-1]
                            page = document.metadata['page']+1
                            retrieval_rank = document.metadata.get('retrieval_rank', 999)  # Default high rank if missing
                            score = document.metadata.get('similarity_score', 0.0)  # Get score if available
                            
                            ranked_docs.append({
                                'file': file,
                                'page': page,
                                'rank': retrieval_rank,
                                'score': score
                            })
                        
                        ranked_docs.sort(key=lambda x: x['rank'])
                        
                        document_name = [doc['file'] for doc in ranked_docs]
                        page_number = [doc['page'] for doc in ranked_docs]
                        
                        results = {
                            'response': result['result'],
                            'documents': document_name,
                            'page_numbers': page_number,
                        }
            return results
                
