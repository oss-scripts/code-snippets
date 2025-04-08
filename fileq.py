@app.post('/query-files')
async def query_files(request: FileQueryRequest):
    try:
        # Extract data from request
        file_info = request.meta_list
        query = request.query.strip()
        
        # Validate input data
        if not file_info:
            return JSONResponse(content={"response": "No file metadata provided."})
        
        # Use the dedicated processing function
        response = process_file_metadata_query(query, file_info)
        
        # Return just the response string as requested
        return JSONResponse(content={"response": response})
        
    except Exception as e:
        import traceback
        print(f"Error processing file query: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )

# Add this function alongside your other utility functions

def create_file_query_prompt():
    """Create a prompt template for file metadata queries."""
    template = """You are a precise file metadata assistant. Your task is to analyze file information and answer questions about it.

CONTEXT INFORMATION:
{file_metadata}

USER QUERY: {query}

INSTRUCTIONS:
1. Answer ONLY based on the file metadata provided above
2. Provide specific details in your answer (exact counts, file names, uploader names, etc.)
3. If asked about files by a specific uploader, count ONLY files from that person
4. For 'how many' questions, always include the total count and a brief listing of files
5. Format lists with bullet points for readability
6. If asked about topics/subjects, look for relevant keywords in filenames
7. Provide direct, factual answers - be concise and specific
8. Do NOT make up any information or files not listed in the metadata

YOUR RESPONSE:"""

    return PromptTemplate(
        template=template, 
        input_variables=["file_metadata", "query"]
    )

def process_file_metadata_query(query, file_info):
    """Process file metadata queries using a dedicated LLM chain."""
    try:
        # Format the metadata into a readable format
        formatted_metadata = "File Metadata:\n"
        for i, file in enumerate(file_info, 1):
            file_name = file.get("Document Name", file.get("name", "Unknown"))
            category = file.get("Category", file.get("category", "Unknown"))
            uploader = file.get("Uploaded By", file.get("uploader", "Unknown"))
            upload_time = file.get("Time of Upload", file.get("upload_time", "Unknown"))
            
            formatted_metadata += f"{i}. Filename: {file_name}, Category: {category}, "
            formatted_metadata += f"Uploaded By: {uploader}, Upload Time: {upload_time}\n"
        
        # Create the LLM chain
        file_query_prompt = create_file_query_prompt()
        llm = LLama3LLM(temperature=0.1, max_tokens=1024)  # Lower temperature for more factual responses
        file_query_chain = LLMChain(
            llm=llm,
            prompt=file_query_prompt,
            verbose=True  # Set to False in production
        )
        
        # Execute the chain
        response = file_query_chain.run({
            "file_metadata": formatted_metadata,
            "query": query
        })
        
        # Post-process the response
        return response.strip()
        
    except Exception as e:
        import traceback
        print(f"Error in file metadata query processing: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return f"Error processing file query: {str(e)}"
