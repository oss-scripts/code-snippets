def retrieve_docs_faiss_followup(query, embedding, is_arabic, chat_history, category):
    print("Is the question in arabic language ", is_arabic)
    print("Filter on category :", category)
    print("**********************")
    print("CHAT HISTORY", chat_history)
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

    # Create a custom retriever that preserves relevance scores
    if category == 'Admin':
        retriever = vectorstore_faiss.as_retriever(
            search_type='similarity', 
            search_kwargs={"k": 7},
            return_source_documents=True
        )
    else:
        retriever = vectorstore_faiss.as_retriever(
            search_type='similarity', 
            search_kwargs={"k": 7, "filter": {'Category': category}},
            return_source_documents=True
        )
    
    # Create QA chain with the retriever
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"verbose": True, "prompt": prompt}
    )
            
    def answer_question(question, chat_history):
        result = {}
        
        if is_arabic:
            translated_question = translate_arabic_question(question)
            print("****************************************")
            print('TRANSLATED QUESTION', translated_question)
            print("****************************************")

            query_type = classify_query_with_llm(translated_question, classifier_llm)
            
            if query_type == "file_metadata":
                answer, docs = process_file_query(translated_question, category, classifier_llm)
                final_answer = refine_arabic_answer(answer)
                results = {'response': final_answer, 'documents': docs, 'page_numbers': [1] * len(docs)}
                return results

            if(classify_question(translated_question)=="Yes"):
                result['result'] = "Hi I am a QnA bot, please feel free to ask a relevant question."
                results={'response':result['result'],'documents':[],'page_numbers':[]}
            else:
                # Get raw results from the vectorstore to capture relevance scores
                raw_docs_with_scores = vectorstore_faiss.similarity_search_with_score(
                    translated_question, 
                    k=7,
                    filter={'Category': category} if category != 'Admin' else None
                )
                
                # Sort by relevance score (lower is better in FAISS with cosine similarity)
                sorted_docs_with_scores = sorted(raw_docs_with_scores, key=lambda x: x[1])
                
                # Run the QA chain
                result = qa({"query": translated_question})
                
                if(("I don't know, please ask another question" in result['result']) or ("I apologize" in result['result'])):
                    results = {'response': result['result'], 'documents': [], 'page_numbers': [], 'relevance_scores': []}
                else:
                    if len(result['source_documents']) == 0:
                        result['result'] = "No information available in the LoB documents."
                        document_name = []
                        page_number = []
                        relevance_scores = []
                    else:
                        # Extract document info in order of relevance
                        document_name = []
                        page_number = []
                        relevance_scores = []
                        
                        # Use the sorted documents from our direct similarity search
                        for doc, score in sorted_docs_with_scores:
                            file = doc.metadata['file_path'].split('/')[-1]
                            page = doc.metadata['page'] + 1
                            document_name.append(file)
                            page_number.append(page)
                            relevance_scores.append(float(score))  # Convert to float for JSON serialization
                
                    results = {
                        'response': result['result'],
                        'documents': document_name,
                        'page_numbers': page_number,
                        'relevance_scores': relevance_scores  # Add relevance scores to response
                    }
                    
            final_answer = refine_arabic_answer(results['response'])
            results['response'] = final_answer
            return results
        else:
            if(classify_question(question)=="Yes"):
                result['result'] = "Hi I am a QnA bot, please feel free to ask a relevant question."
                results = {'response': result['result'], 'documents': [], 'page_numbers': [], 'relevance_scores': []}
            else:
                query_type = classify_query_with_llm(question, classifier_llm)
            
                if query_type == "file_metadata":
                    answer, docs = process_file_query(question, category, classifier_llm)
                    return {'response': answer, 'documents': docs, 'page_numbers': [1] * len(docs), 'relevance_scores': [0] * len(docs)}

                contexualized_question = followup_question([chat_history, question])
                print("-----------------------------------")
                print("this is contexual question")
                print(contexualized_question)
                print("-----------------------------------")
                
                # Get raw results from the vectorstore to capture relevance scores
                raw_docs_with_scores = vectorstore_faiss.similarity_search_with_score(
                    contexualized_question, 
                    k=7,
                    filter={'Category': category} if category != 'Admin' else None
                )
                
                # Sort by relevance score (lower is better with cosine similarity)
                sorted_docs_with_scores = sorted(raw_docs_with_scores, key=lambda x: x[1])
                
                result = qa({"query": contexualized_question})
                
                if(("I don't know, please ask another question" in result['result']) or ("I apologize" in result['result'])):
                    results = {'response': result['result'], 'documents': [], 'page_numbers': [], 'relevance_scores': []}
                else:
                    if len(result['source_documents']) == 0:
                        result['result'] = "No information available in the LoB documents."
                        document_name = []
                        page_number = []
                        relevance_scores = []
                    else:
                        # Extract document info in order of relevance
                        document_name = []
                        page_number = []
                        relevance_scores = []
                        
                        # Use the sorted documents from our direct similarity search
                        for doc, score in sorted_docs_with_scores:
                            file = doc.metadata['file_path'].split('/')[-1]
                            page = doc.metadata['page'] + 1
                            document_name.append(file)
                            page_number.append(page)
                            relevance_scores.append(float(score))  # Convert to float for JSON serialization
                
                    results = {
                        'response': result['result'],
                        'documents': document_name,
                        'page_numbers': page_number,
                        'relevance_scores': relevance_scores  # Add relevance scores to response
                    }
                    
            return results
                
    answer = answer_question(query, chat_history)
    print("********************")
    print(answer)

    return answer
