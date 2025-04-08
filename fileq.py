@app.post('/query-files')
async def query_files(request: FileQueryRequest):
    try:
        # Extract data from request
        file_info = request.meta_list
        query = request.query.strip()
        
        # Log the incoming request for debugging
        print(f"Processing file metadata query: '{query}'")
        print(f"Metadata sample: {file_info[:2] if len(file_info) > 2 else file_info}")
        
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

def create_file_query_prompt():
    """Create a prompt template for file metadata queries."""
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a specialized file metadata assistant for banking documents.
Your task is to analyze the provided file information and answer questions about it accurately.

FILE METADATA:
{file_metadata}

IMPORTANT INSTRUCTIONS:
1. Answer ONLY based on the file metadata provided above
2. Provide specific details (exact counts, file names, uploader names, etc.)
3. If asked about files by a specific uploader, count ONLY files from that person
4. For 'how many' questions, include the total count and briefly list the files
5. Format lists with bullet points for readability
6. If asked about topics/subjects, look for relevant keywords in filenames
7. Provide direct, factual answers - be concise and specific
8. Never make up any information not present in the metadata
9. If you cannot answer based on the given metadata, say "I cannot determine this from the available metadata"
10. Keep your answers professional and relevant to banking document management
<|eot_id|>
<|start_header_id|>user<|end_header_id|>Question:{query}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n"""

    return PromptTemplate(
        template=template, 
        input_variables=["file_metadata", "query"]
    )

def process_file_metadata_query(query, file_info):
    """Process file metadata queries using a dedicated LLM chain with properly formatted prompts."""
    try:
        # Format the metadata into a readable format
        formatted_metadata = ""
        for i, file in enumerate(file_info, 1):
            file_name = file.get("Document Name", file.get("name", "Unknown"))
            category = file.get("Category", file.get("category", "Unknown"))
            uploader = file.get("Uploaded By", file.get("uploader", "Unknown"))
            upload_time = file.get("Time of Upload", file.get("upload_time", "Unknown"))
            
            formatted_metadata += f"File {i}:\n"
            formatted_metadata += f"- Name: {file_name}\n"
            formatted_metadata += f"- Category: {category}\n"
            formatted_metadata += f"- Uploaded By: {uploader}\n"
            formatted_metadata += f"- Upload Time: {upload_time}\n\n"
        
        # Create the LLM chain
        file_query_prompt = create_file_query_prompt()
        llm = LLama3LLM()  # Use your existing LLama3LLM class
        
        # Create and run the chain
        file_query_chain = LLMChain(
            llm=llm,
            prompt=file_query_prompt,
            verbose=False  # Can be set to True for debugging
        )
        
        # Execute the chain
        response = file_query_chain.run({
            "file_metadata": formatted_metadata,
            "query": query
        })
        
        # Return the processed response
        return response.strip()
        
    except Exception as e:
        import traceback
        print(f"Error in file metadata query processing: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return f"Error processing file query: {str(e)}"
