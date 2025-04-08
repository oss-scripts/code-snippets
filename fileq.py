@app.post('/query-files')
async def query_files(request: FileQueryRequest):
    try:
        file_info = request.meta_list
        query = request.query.strip()
        
        formatted_metadata = "File Metadata:\n"
        for i, file in enumerate(file_info, 1):
            file_name = file.get("Document Name", file.get("name", "Unknown"))
            category = file.get("Category", file.get("category", "Unknown"))
            uploader = file.get("Uploaded By", file.get("uploader", "Unknown"))
            upload_time = file.get("Time of Upload", file.get("upload_time", "Unknown"))
            
            formatted_metadata += f"{i}. Filename: {file_name}, Category: {category}, "
            formatted_metadata += f"Uploaded By: {uploader}, Upload Time: {upload_time}\n"
        
        llm_prompt = f"""You are an assistant that answers questions about file metadata. 
Below is a list of files with their metadata:

{formatted_metadata}

User query: {query}

Provide a clear, concise answer to the query based only on the file metadata above. 
Keep your response factual and to the point. Include specific details like counts, 
filenames, and other relevant information directly from the metadata.

Your response:"""
        
        llm = LLama3LLM()
        
        llm_response = llm(llm_prompt)
        
        response = llm_response.strip()
        
        return JSONResponse(content={"response": response})
        
    except Exception as e:
        import traceback
        print(f"Error processing file query: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )
