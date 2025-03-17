def classify_query_with_llm(question, llm):
     """Use LLM to determine if a query is about file metadata or document content."""
     classification_prompt = PromptTemplate(
         template="""You are a query classifier.
 Determine if the following query is asking about file metadata/statistics or document content.
 
 Query: "{question}"
 
 If the query is asking about files themselves (like how many files, listing files, finding files by name/category, etc.), 
 classify it as "FILE_METADATA".
 
 If the query is asking about information contained within documents, classify it as "DOCUMENT_CONTENT".
 
 Your classification (respond with only "FILE_METADATA" or "DOCUMENT_CONTENT"):""",
         input_variables=["question"]
     )
     
     classification_chain = LLMChain(llm=llm, prompt=classification_prompt)
     classification = classification_chain.run(question).strip()
     
     if "FILE_METADATA" in classification.upper():
         return "file_metadata"
     else:
         return "document_content"
 
 def process_file_query(query, category, llm):
     """Process queries about file metadata and statistics using LLM."""
     registry = load_file_registry()
     
     # Apply category filter if specified
     if category and category != "ALL" and category != "Admin":
         filtered_registry = {
             k: v for k, v in registry.items() 
             if v.get("category") == category
         }
     else:
         filtered_registry = registry
     
     # Gather file statistics
     total_files = len(filtered_registry)
     categories = Counter([info.get("category", "Unknown") for info in filtered_registry.values()])
     
     # Create a structured context with file information
     file_context = f"""File Statistics:
 Total files: {total_files}
 
 Files by category:
 {', '.join([f'{cat}: {count}' for cat, count in categories.items()])}
 
 """
     
     # Always include complete file listing to enable detailed responses
     file_context += "\nComplete file listing:\n"
     for key, info in filtered_registry.items():
         file_context += f"- {info.get('name', 'Unknown')} (Category: {info.get('category', 'Unknown')})\n"
     
     # Create an enhanced prompt for the LLM to answer the file query with detailed information
     file_answer_prompt = PromptTemplate(
         template="""You are a helpful assistant providing information about the document collection.
 Use the following file statistics to answer the user's question.
 
 {file_context}
 
 User question: {question}
 
 Answer the question accurately and helpfully based on the file statistics provided.
 If the question asks about how many files are available, always specify the exact number and list the actual file names as bullet points.
 If the question asks about specific files or categories, provide the full list of relevant files.
 If the query is about files in a specific category, list all files in that category.
 If there are more than 15 relevant files, list only the first 15 and mention how many more are available.
 If the question asks about specific files or categories not present in the data, explain what IS available instead.
 
 IMPORTANT: Do not use placeholders like [number] or [category] in your answer. Always use the actual values from the file statistics.
 
 Your answer should be detailed but well-organized with proper headings and bullet points where appropriate.
 
 Your answer:""",
         input_variables=["file_context", "question"]
     )
     
     # Get LLM to generate the answer
     file_answer_chain = LLMChain(llm=llm, prompt=file_answer_prompt)
     answer = file_answer_chain.run(file_context=file_context, question=query)
     
     # Post-process the response to ensure no placeholders remain
     answer = answer.replace("[number]", str(total_files))
     
     # If we detect a potential placeholder pattern, try to post-process it
     if "[" in answer and "]" in answer:
         for cat in categories.keys():
             answer = answer.replace(f"[{cat}]", str(categories.get(cat, 0)))
             answer = answer.replace(f"[{cat.lower()}]", str(categories.get(cat, 0)))
             answer = answer.replace(f"[{cat.upper()}]", str(categories.get(cat, 0)))
     
     # Find relevant file names to return as "sources"
     document_names = []
     
     # Extract keywords from query to find potentially relevant files
     keywords = [word.lower() for word in query.split() 
                if len(word) > 3 and word.lower() not in ["files", "documents", "related", "about", "with", "have", "many", "show", "list"]]
     
     # If asking about a specific category, include all files from that category as sources
     category_match = re.search(r'(in|from|about)\s+(\w+)\s+(category|department)', query, re.IGNORECASE)
     if category_match:
         requested_category = category_match.group(2).upper()
         for key, info in filtered_registry.items():
             if info.get("category", "").upper() == requested_category:
                 document_names.append(info.get("name", "Unknown"))
     # Otherwise use keyword matching
     elif keywords:
         for key, info in filtered_registry.items():
             file_name = info.get("name", "").lower()
             if any(keyword in file_name for keyword in keywords):
                 document_names.append(info.get("name", "Unknown"))
     # If just asking about files in general, include some representative files
     else:
         for key, info in list(filtered_registry.items())[:10]:  # First 10 files as examples
             document_names.append(info.get("name", "Unknown"))
     
     # Limit to most relevant files
     document_names = list(set(document_names))[:10]  # Unique, limited to 10
     
     return answer, document_names





"""

            query_type = classify_query_with_llm(translated_question, classifier_llm)
             
             if query_type == "file_metadata":
                 answer, docs = process_file_query(translated_question, category, classifier_llm)
                 final_answer = refine_arabic_answer(answer)
                 results = {'response': final_answer, 'documents': docs, 'page_numbers': [1] * len(docs)}
                 return results
 
 
"""

"""
                query_type = classify_query_with_llm(question, classifier_llm)
             
                 if query_type == "file_metadata":
                     answer, docs = process_file_query(question, category, classifier_llm)
                     return {'response': answer, 'documents': docs, 'page_numbers': [1] * len(docs)}
 

"""
