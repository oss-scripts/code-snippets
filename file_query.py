# ...existing code...

class FileQueryRequest(BaseModel):
    query: str
    meta_list: List[Dict[str, Any]]

@app.post('/query-files')
async def query_files(request: FileQueryRequest):
    try:
        # Use the file metadata provided in the payload
        file_info = request.meta_list
        query = request.query.lower()
        
        # Convert all timestamps to datetime objects for comparison
        from datetime import datetime, timedelta
        import dateutil.parser
        
        # Process dates in file metadata
        for file in file_info:
            if "Time of Upload" in file:
                try:
                    file["upload_datetime"] = dateutil.parser.parse(file["Time of Upload"])
                except:
                    # If parsing fails, set a default old date
                    file["upload_datetime"] = datetime(2000, 1, 1)
        
        # Check if we can answer with direct logic before using LLM
        # Get current date for time-based comparisons
        current_date = datetime.now()
        
        # Handle user-based queries directly
        user_match = re.search(r"(?:uploaded|added) by\s+['\"]*(\w+(?:\.\w+)*)['\"]*", query)
        if user_match:
            username = user_match.group(1)
            matching_files = [f for f in file_info if f.get("Uploaded By", "").lower() == username.lower()]
            
            return JSONResponse(content={
                "response": f"Found {len(matching_files)} files uploaded by {username}.",
                "files": [f.get("Document Name") for f in matching_files],
                "count": len(matching_files)
            })
        
        # Handle time-based queries directly
        time_patterns = {
            "last week": timedelta(days=7),
            "last month": timedelta(days=30),
            "last year": timedelta(days=365),
        }
        
        # Check for "last n days/weeks/months" pattern
        time_match = re.search(r"last\s+(\d+)\s+(day|days|week|weeks|month|months)", query)
        if time_match:
            number = int(time_match.group(1))
            unit = time_match.group(2)
            
            if unit in ["day", "days"]:
                delta = timedelta(days=number)
            elif unit in ["week", "weeks"]:
                delta = timedelta(days=number * 7)
            elif unit in ["month", "months"]:
                delta = timedelta(days=number * 30)
                
            since_date = current_date - delta
            matching_files = [f for f in file_info if f.get("upload_datetime", datetime(2000, 1, 1)) >= since_date]
            
            return JSONResponse(content={
                "response": f"Found {len(matching_files)} files uploaded in the last {number} {unit}.",
                "files": [f.get("Document Name") for f in matching_files],
                "count": len(matching_files)
            })
        
        # Check for predefined time periods
        for time_key, delta in time_patterns.items():
            if time_key in query:
                since_date = current_date - delta
                matching_files = [f for f in file_info if f.get("upload_datetime", datetime(2000, 1, 1)) >= since_date]
                
                return JSONResponse(content={
                    "response": f"Found {len(matching_files)} files uploaded in {time_key}.",
                    "files": [f.get("Document Name") for f in matching_files],
                    "count": len(matching_files)
                })
        
        # Check for "after specific date" pattern
        date_match = re.search(r"after\s+(\d{4}-\d{2}-\d{2})", query)
        if date_match:
            date_str = date_match.group(1)
            try:
                since_date = datetime.strptime(date_str, "%Y-%m-%d")
                matching_files = [f for f in file_info if f.get("upload_datetime", datetime(2000, 1, 1)) >= since_date]
                
                return JSONResponse(content={
                    "response": f"Found {len(matching_files)} files uploaded after {date_str}.",
                    "files": [f.get("Document Name") for f in matching_files],
                    "count": len(matching_files)
                })
            except:
                pass  # If date parsing fails, continue to LLM
        
        # If no direct patterns match, use LLM for more complex queries
        # Prepare the prompt for the LLM
        prompt = f"""
        You are a file query assistant. Answer questions about the following files:
        
        {json.dumps(file_info, indent=2)}
        
        User query: {request.query}
        
        Provide a direct answer to the query. Answer in these specific ways:
        
        1. If the question is about counting files, give the exact count.
        
        2. If the question is about finding files related to a specific topic,
           identify the relevant files by analyzing the document names.
           
        3. If the question is about files uploaded by a specific user,
           count and list the files uploaded by that user.
           
        4. If the question is about files uploaded during a specific time period
           (like last week, last month, or after a specific date),
           identify those files based on the "Time of Upload" field.
        
        Format your answer as a concise and helpful response.
        """
        
        # Process with LLM
        llm = LLama3LLM()
        answer = llm(prompt)
        
        # Prepare a structured response
        response = {
            "response": answer,
            "file_count": len(file_info)
        }
        
        # Try to extract file names mentioned in the response
        try:
            file_names = [file["Document Name"] for file in file_info]
            mentioned_files = [f for f in file_names if f in answer]
            if mentioned_files:
                response["files"] = mentioned_files
        except Exception:
            # If extraction fails, just continue with the general response
            pass
                
        return JSONResponse(content=response)
            
    except Exception as e:
        print(f"Error processing file query: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )
