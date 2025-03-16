class DeleteFileRequest(BaseModel):
    file_keys: Union[str, List[str]]
    categories: Union[str, List[str]]

@app.post('/delete-file')
async def delete_file(request: DeleteFileRequest):
    try:
        file_keys = [request.file_keys] if isinstance(request.file_keys, str) else request.file_keys
        categories = [request.categories] if isinstance(request.categories, str) else request.categories
        
        if len(categories) < len(file_keys):
            categories.extend([None] * (len(file_keys) - len(categories)))
            
        results = []
        registry = load_file_registry()
        modified_registry = False
        
        for i, file_key in enumerate(file_keys):
            if file_key not in registry:
                results.append({
                    "file_key": file_key,
                    "success": False,
                    "message": f"No file found with key: {file_key}"
                })
                continue
            
            file_info = registry[file_key]
            category = categories[i] if i < len(categories) else None
            
            if category and file_info.get("category") != category:
                results.append({
                    "file_key": file_key, 
                    "success": False,
                    "message": f"File does not belong to category: {category}"
                })
                continue
                
            file_path = file_info["path"]
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    results.append({
                        "file_key": file_key,
                        "success": False,
                        "message": f"Error deleting file: {str(e)}"
                    })
                    continue
            
            registry.pop(file_key)
            modified_registry = True
            
            if os.path.exists(index_path):
                success = delete_documents_by_file_key(vectorstore_faiss, file_key)
                if success:
                    print(f"Updated index: Removed documents with file_key: {file_key}")
                    results.append({
                        "file_key": file_key,
                        "success": True,
                        "message": "File deleted successfully"
                    })
                else:
                    results.append({
                        "file_key": file_key,
                        "success": True,
                        "message": "File deleted but no documents found in index"
                    })
            else:
                results.append({
                    "file_key": file_key,
                    "success": True,
                    "message": "File deleted (no index found)"
                })
        
        if modified_registry:
            save_file_registry(registry)
        
        successful = sum(1 for r in results if r["success"])
        
        if len(file_keys) == 1:
            result = results[0]
            if result["success"]:
                return JSONResponse(content={'response': result["message"]})
            else:
                return JSONResponse(
                    status_code=400 if "not found" in result["message"] else 500,
                    content={'error': result["message"]}
                )
        else:
            return JSONResponse(content={
                'response': f"Successfully deleted {successful} of {len(file_keys)} files",
                'results': results
            })
            
    except Exception as e:
        print(f"Error during file deletion: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )
