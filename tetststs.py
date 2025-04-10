def process_file_metadata_query(query, file_info):
    """Process file metadata queries using pandas operations to support LLM responses."""
    try:
        # Convert the file metadata to a pandas DataFrame
        df = pd.DataFrame.from_dict(file_info)
        
        # Normalize column names (handle case variations)
        df.columns = [col.strip() for col in df.columns]
        column_mapping = {
            'document name': 'Document Name',
            'uploaded by': 'Uploaded By',
            'time of upload': 'Time of Upload',
            'category': 'Category'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Make sure we have all expected columns with default values if missing
        for col in ['Document Name', 'Uploaded By', 'Time of Upload', 'Category']:
            if col not in df.columns:
                df[col] = "Unknown"
        
        # Clean and standardize data
        if 'Document Name' in df.columns:
            df['Document Name'] = df['Document Name'].str.strip()
        
        # Convert dates to datetime format for proper querying
        if 'Time of Upload' in df.columns:
            df['Time of Upload'] = pd.to_datetime(df['Time of Upload'], errors='coerce')
            df['Upload Month'] = df['Time of Upload'].dt.month_name()
            df['Upload Year'] = df['Time of Upload'].dt.year
            df['Upload Date'] = df['Time of Upload'].dt.date
        
        # Pre-compute answers to common question types to help the LLM
        precomputed_answers = {}
        
        # 1. Get total file count
        total_files = len(df)
        precomputed_answers['total_files'] = f"There are {total_files} files in the database."
        
        # 2. Files by category
        if 'Category' in df.columns:
            category_counts = df['Category'].value_counts().to_dict()
            category_summary = "\n".join([f"- {cat}: {count} files" for cat, count in category_counts.items()])
            precomputed_answers['files_by_category'] = f"Files by category:\n{category_summary}"
        
        # 3. Files by uploader
        if 'Uploaded By' in df.columns:
            uploader_counts = df['Uploaded By'].value_counts().to_dict()
            uploader_summary = "\n".join([f"- {uploader}: {count} files" for uploader, count in uploader_counts.items()])
            precomputed_answers['files_by_uploader'] = f"Files by uploader:\n{uploader_summary}"
        
        # 4. Get all uploader names
        if 'Uploaded By' in df.columns:
            uploaders = df['Uploaded By'].unique().tolist()
            precomputed_answers['all_uploaders'] = f"Uploaders: {', '.join(uploaders)}"
        
        # 5. Most recent files (last 5)
        if 'Time of Upload' in df.columns and not df['Time of Upload'].isna().all():
            recent_files = df.sort_values('Time of Upload', ascending=False).head(5)
            recent_file_list = "\n".join([
                f"- {row['Document Name']} (uploaded by {row['Uploaded By']} on {row['Upload Date']})" 
                for _, row in recent_files.iterrows()
            ])
            precomputed_answers['recent_files'] = f"Most recent files:\n{recent_file_list}"
        
        # Check for specific query patterns and add targeted information
        query_lower = query.lower()
        additional_context = []
        
        # A. Check for uploader-specific queries
        if any(term in query_lower for term in ["uploaded by", "files by", "from user"]):
            # Extract potential uploader names
            if 'Uploaded By' in df.columns:
                for uploader in df['Uploaded By'].unique():
                    if uploader.lower() in query_lower:
                        uploader_files = df[df['Uploaded By'] == uploader]
                        file_list = "\n".join([f"- {row['Document Name']}" for _, row in uploader_files.iterrows()])
                        additional_context.append(
                            f"Files uploaded by {uploader} ({len(uploader_files)} total):\n{file_list}"
                        )
                        break
        
        # B. Check for keyword/content queries
        if any(term in query_lower for term in ["about", "containing", "with", "related to"]):
            # Extract keywords (words longer than 3 chars that aren't common filler words)
            words = query_lower.split()
            stop_words = ["the", "and", "files", "documents", "about", "containing", "with", "related", "to", "for", "are", "have"]
            potential_keywords = [word for word in words if len(word) > 3 and word not in stop_words]
            
            if potential_keywords and 'Document Name' in df.columns:
                # Create a mask for documents containing any of these keywords
                mask = df['Document Name'].str.lower().apply(
                    lambda name: any(keyword in name for keyword in potential_keywords)
                )
                matching_files = df[mask]
                
                if not matching_files.empty:
                    file_list = "\n".join([f"- {row['Document Name']}" for _, row in matching_files.iterrows()])
                    keywords_str = ", ".join(potential_keywords)
                    additional_context.append(
                        f"Files matching keywords '{keywords_str}' ({len(matching_files)} total):\n{file_list}"
                    )
        
        # C. Check for date/time queries
        if any(term in query_lower for term in ["recent", "latest", "newest", "month", "year", "date", "time"]):
            if 'Time of Upload' in df.columns and not df['Time of Upload'].isna().all():
                # Check for month queries
                months = ["january", "february", "march", "april", "may", "june", "july", 
                          "august", "september", "october", "november", "december"]
                
                for month in months:
                    if month in query_lower and 'Upload Month' in df.columns:
                        month_files = df[df['Upload Month'].str.lower() == month]
                        if not month_files.empty:
                            file_list = "\n".join([f"- {row['Document Name']}" for _, row in month_files.iterrows()])
                            additional_context.append(
                                f"Files uploaded in {month.capitalize()} ({len(month_files)} total):\n{file_list}"
                            )
                            break
                
                # Check for year queries
                years = [str(year) for year in range(2020, 2026)]  # Adjust range as needed
                for year in years:
                    if year in query_lower and 'Upload Year' in df.columns:
                        year_files = df[df['Upload Year'] == int(year)]
                        if not year_files.empty:
                            file_list = "\n".join([f"- {row['Document Name']}" for _, row in year_files.iterrows()])
                            additional_context.append(
                                f"Files uploaded in {year} ({len(year_files)} total):\n{file_list}"
                            )
                            break
        
        # Combine precomputed answers with any additional context found
        pandas_analysis = "\n\n".join([
            "DATABASE ANALYSIS (Based on direct data processing):",
            precomputed_answers['total_files'],
            precomputed_answers.get('files_by_category', ''),
            precomputed_answers.get('files_by_uploader', ''),
        ])
        
        if additional_context:
            pandas_analysis += "\n\nRELEVANT QUERY RESULTS:\n" + "\n\n".join(additional_context)
        
        # Convert the original df to CSV for completeness
        csv_string = df.to_csv(None, index=False)
        
        # Create the LLM chain with enhanced prompt
        file_query_prompt = create_file_query_prompt()
        llm = LLama3LLM()
        
        # Provide both the structured analysis and the raw data to the LLM
        file_query_chain = LLMChain(
            llm=llm,
            prompt=file_query_prompt,
            verbose=False
        )
        
        # Execute the chain with enhanced metadata
        enhanced_metadata = f"{pandas_analysis}\n\nRAW DATA (CSV Format):\n{csv_string}"
        response = file_query_chain.run({
            "file_metadata": enhanced_metadata,
            "query": query
        })
        
        # Return the processed response
        return response.strip()
        
    except Exception as e:
        import traceback
        print(f"Error in file metadata query processing: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return f"Error processing file query: {str(e)}"
