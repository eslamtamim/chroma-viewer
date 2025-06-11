import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd 
import streamlit as st
import argparse

parser = argparse.ArgumentParser(description='View Chroma DB collections')
parser.add_argument('--host', default='localhost', help='ChromaDB host (for Docker/server mode)')
parser.add_argument('--port', type=int, default=8000, help='ChromaDB port (for Docker/server mode)')
parser.add_argument('--path', help='Local ChromaDB path (for persistent mode)')

pd.set_option('display.max_columns', 4)

def view_collections_http(host, port):
    st.markdown("### ChromaDB Server: %s:%s" % (host, port))
    
    try:
        client = chromadb.HttpClient(host=host, port=port)
        display_collections(client)
    except Exception as e:
        st.error(f"Failed to connect to ChromaDB server at {host}:{port}")
        st.error(f"Error: {str(e)}")
        st.info("Make sure your ChromaDB Docker container is running and accessible")

def view_collections_persistent(path):
    st.markdown("### DB Path: %s" % path)
    
    try:
        client = chromadb.PersistentClient(path=path)
        display_collections(client)
    except Exception as e:
        st.error(f"Failed to open ChromaDB at path: {path}")
        st.error(f"Error: {str(e)}")

def display_collections(client):
    # This might take a while in the first execution if Chroma wants to download
    # the embedding transformer
    collections = client.list_collections()
    print(f"Found {len(collections)} collections")

    st.header("Collections")
    
    if not collections:
        st.info("No collections found in this ChromaDB instance")
        return

    for collection in collections:
        try:
            data = collection.get()

            ids = data['ids']
            embeddings = data["embeddings"]
            metadata = data["metadatas"]
            documents = data["documents"]

            st.markdown("### Collection: **%s**" % collection.name)
            st.markdown(f"**Count:** {len(ids)} items")
            
            # Check if arrays have the same length
            lengths = {
                'ids': len(ids) if ids else 0,
                'embeddings': len(embeddings) if embeddings else 0,
                'metadatas': len(metadata) if metadata else 0,
                'documents': len(documents) if documents else 0
            }
            
            st.write(f"Data lengths: {lengths}")
            
            # Create DataFrame only with non-empty arrays of the same length
            max_length = max(lengths.values())
            if max_length == 0:
                st.info("This collection is empty")
                continue
                
            # Create a clean DataFrame by ensuring all arrays have the same length
            df_data = {}
            if ids and len(ids) == max_length:
                df_data['ids'] = ids
            if documents and len(documents) == max_length:
                df_data['documents'] = documents
            if metadata and len(metadata) == max_length:
                df_data['metadatas'] = metadata
            # Note: embeddings are usually very long arrays, so we'll show their count instead
            if embeddings and len(embeddings) == max_length:
                df_data['embeddings_count'] = [len(emb) if emb else 0 for emb in embeddings]

            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df)
                
                # Add query and browse functionality
                tab1, tab2 = st.tabs(["Query Collection", "Browse Collection"])
                
                with tab1:
                    st.markdown("#### Search Collection")
                    
                    # Query input
                    query_text = st.text_input(f"Search in '{collection.name}':", key=f"query_{collection.name}")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        n_results = st.number_input("Number of results:", min_value=1, max_value=50, value=5, key=f"n_results_{collection.name}")
                    
                    with col2:
                        include_distances = st.checkbox("Include distances", value=True, key=f"distances_{collection.name}")
                    
                    # Advanced options
                    with st.expander("Advanced Query Options"):
                        where_filter = st.text_input(
                            "Metadata filter (JSON format, e.g., {'key': 'value'}):", 
                            key=f"where_{collection.name}",
                            help="Filter results by metadata. Use JSON format like {'category': 'science'}"
                        )
                        where_document_filter = st.text_input(
                            "Document content filter (e.g., {'$contains': 'keyword'}):", 
                            key=f"where_doc_{collection.name}",
                            help="Filter by document content. Use operators like {'$contains': 'word'}"
                        )
                
                with tab2:
                    st.markdown("#### Browse Collection")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        browse_limit = st.number_input("Items to show:", min_value=1, max_value=100, value=10, key=f"browse_limit_{collection.name}")
                    with col2:
                        browse_offset = st.number_input("Start from item:", min_value=0, value=0, key=f"browse_offset_{collection.name}")
                    with col3:
                        if st.button("Browse", key=f"browse_btn_{collection.name}"):
                            try:
                                browse_results = collection.get(
                                    limit=browse_limit,
                                    offset=browse_offset
                                )
                                
                                st.markdown("##### Browse Results:")
                                if browse_results['ids']:
                                    browse_data = {}
                                    if browse_results['ids']:
                                        browse_data['ids'] = browse_results['ids']
                                    if browse_results['documents']:
                                        browse_data['documents'] = browse_results['documents']
                                    if browse_results['metadatas']:
                                        browse_data['metadatas'] = browse_results['metadatas']
                                    
                                    if browse_data:
                                        browse_df = pd.DataFrame(browse_data)
                                        st.dataframe(browse_df)
                                        st.write(f"Showing items {browse_offset} to {browse_offset + len(browse_results['ids'])}")
                                    else:
                                        st.info("No data to display")
                                else:
                                    st.info("No items found in the specified range")
                            except Exception as browse_error:
                                st.error(f"Browse failed: {str(browse_error)}")
                
                if query_text:
                    try:
                        # Parse filters
                        where_dict = None
                        where_document_dict = None
                        
                        if where_filter:
                            try:
                                import json
                                where_dict = json.loads(where_filter)
                            except json.JSONDecodeError:
                                st.warning("Invalid JSON format in metadata filter. Ignoring filter.")
                        
                        if where_document_filter:
                            try:
                                import json
                                where_document_dict = json.loads(where_document_filter)
                            except json.JSONDecodeError:
                                st.warning("Invalid JSON format in document filter. Ignoring filter.")
                        
                        # Perform the query
                        query_params = {
                            "query_texts": [query_text],
                            "n_results": min(n_results, max_length)  # Don't exceed available items
                        }
                        
                        if where_dict:
                            query_params["where"] = where_dict
                        
                        if where_document_dict:
                            query_params["where_document"] = where_document_dict
                        
                        query_results = collection.query(**query_params)
                        
                        st.markdown("##### Query Results:")
                        
                        # Display query results
                        if query_results['ids'] and query_results['ids'][0]:
                            result_data = {}
                            
                            if query_results['ids'][0]:
                                result_data['ids'] = query_results['ids'][0]
                            
                            if query_results['documents'] and query_results['documents'][0]:
                                result_data['documents'] = query_results['documents'][0]
                            
                            if query_results['metadatas'] and query_results['metadatas'][0]:
                                result_data['metadatas'] = query_results['metadatas'][0]
                            
                            if include_distances and query_results['distances'] and query_results['distances'][0]:
                                result_data['distances'] = query_results['distances'][0]
                            
                            if result_data:
                                results_df = pd.DataFrame(result_data)
                                st.dataframe(results_df)
                                
                                # Show some stats
                                if include_distances and 'distances' in result_data:
                                    st.write(f"**Best match distance:** {min(result_data['distances']):.4f}")
                                    st.write(f"**Average distance:** {sum(result_data['distances'])/len(result_data['distances']):.4f}")
                            else:
                                st.info("Query executed but no results to display")
                        else:
                            st.info("No results found for your query")
                            
                    except Exception as query_error:
                        st.error(f"Query failed: {str(query_error)}")
                        st.info("Make sure the collection has embeddings and the query is valid")
                
            else:
                st.warning("Could not create DataFrame due to inconsistent data lengths")
                
        except Exception as e:
            st.error(f"Error processing collection '{collection.name}': {str(e)}")
            continue

if __name__ == "__main__":
    try:
        args = parser.parse_args()
        
        if args.path:
            # Use persistent client for local database
            print("Opening local database: %s" % args.path)
            view_collections_persistent(args.path)
        else:
            # Use HTTP client for Docker/server mode
            print("Connecting to ChromaDB server: %s:%s" % (args.host, args.port))
            view_collections_http(args.host, args.port)
    except Exception as e:
        st.error(f"Application error: {str(e)}")

