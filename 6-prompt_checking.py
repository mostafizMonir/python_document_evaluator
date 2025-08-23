import os
import lancedb
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from pathlib import Path
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


# Initialize LanceDB connection
@st.cache_resource
def init_db():
    """Initialize database connection.

    Returns:
        LanceDB table object
    """
    db = lancedb.connect("data/lancedb")
    return db.open_table("docling")


def get_context(query: str, table, num_results: int = 5) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question or prompt
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    results = table.search(query).limit(num_results).to_pandas()
    contexts = []

    for _, row in results.iterrows():
        # Extract metadata
        filename = row["metadata"]["filename"]
        page_numbers = row["metadata"]["page_numbers"]
        title = row["metadata"]["title"]

        # Build source citation
        source_parts = []
        if filename:
            source_parts.append(filename)
        if page_numbers is not None and len(page_numbers) > 0:
            source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")

        source = f"\nSource: {' - '.join(source_parts)}"
        if title:
            source += f"\nTitle: {title}"

        contexts.append(f"{row['text']}{source}")

    return "\n\n".join(contexts)


def get_llm_response(prompt: str, context: str) -> str:
    """Get response from OpenAI API for a given prompt with context.

    Args:
        prompt: The evaluation prompt
        context: Retrieved context from database

    Returns:
        str: Model's response
    """
    system_prompt = f"""You are a professional grant proposal evaluator. 
    Use the provided document context to evaluate according to the given criteria.
    Base your evaluation only on the information available in the context.
    
    Document Context:
    {context}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,  # Lower temperature for more consistent evaluation
    )

    return response.choices[0].message.content


def process_prompt_files(prompts_folder: str, table):
    """Process all prompt files in the specified folder.

    Args:
        prompts_folder: Path to the folder containing prompt files
        table: LanceDB table object

    Returns:
        dict: Dictionary with prompt filenames as keys and evaluation results as values
    """
    results = {}
    prompt_files = [f for f in os.listdir(prompts_folder) if f.endswith('.txt')]
    
    for prompt_file in prompt_files:
        file_path = os.path.join(prompts_folder, prompt_file)
        
        # Read the prompt content
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        
        # Get relevant context from the database
        # Use the first part of the prompt as the search query
        search_query = prompt_content[:500]  # Use first 500 chars as search query
        context = get_context(search_query, table, num_results=10)  # Get more results for evaluation
        
        # Get LLM response
        response = get_llm_response(prompt_content, context)
        
        results[prompt_file] = {
            'prompt': prompt_content[:200] + '...' if len(prompt_content) > 200 else prompt_content,
            'response': response
        }
    
    return results


# Streamlit UI
st.set_page_config(page_title="Prompt Checker", page_icon="📋", layout="wide")
st.title("📋 Grant Proposal Prompt Checker")
st.markdown("This tool processes evaluation prompts against the document database and generates compliance assessments.")

# Initialize database connection
table = init_db()

# Define prompts folder path
prompts_folder = "../GFA_CFP_Prompts"

# Check if folder exists
if not os.path.exists(prompts_folder):
    st.error(f"Prompts folder not found at: {prompts_folder}")
    st.stop()

# Get list of available prompt files
available_files = [f for f in os.listdir(prompts_folder) if f.endswith('.txt')]
available_files.sort()

if not available_files:
    st.warning("No .txt files found in the prompts folder.")
    st.stop()

# File selection UI
st.markdown("### 📁 Select Files to Process")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    selection_mode = st.radio(
        "Selection Mode:",
        ["All Files", "Multiple Files", "Single File"],
        help="Choose how many files to process"
    )

selected_files = []

if selection_mode == "All Files":
    selected_files = available_files
    with col2:
        st.info(f"📊 Will process all {len(available_files)} files")
    with col3:
        st.markdown("**Files to process:**")
        for f in available_files[:5]:
            st.markdown(f"• {f.replace('.txt', '')}")
        if len(available_files) > 5:
            st.markdown(f"• ... and {len(available_files) - 5} more")

elif selection_mode == "Multiple Files":
    with col2:
        selected_files = st.multiselect(
            "Choose files:",
            available_files,
            default=None,
            format_func=lambda x: x.replace('.txt', ''),
            help="Select multiple files to process"
        )
    with col3:
        if selected_files:
            st.success(f"✅ {len(selected_files)} files selected")
        else:
            st.warning("⚠️ Please select files to process")

elif selection_mode == "Single File":
    with col2:
        selected_file = st.selectbox(
            "Choose a file:",
            [""] + available_files,
            format_func=lambda x: "Select a file..." if x == "" else x.replace('.txt', ''),
            help="Select a single file to process"
        )
        if selected_file:
            selected_files = [selected_file]
    with col3:
        if selected_files:
            st.success(f"✅ File selected: {selected_files[0].replace('.txt', '')}")
        else:
            st.warning("⚠️ Please select a file to process")

st.markdown("---")

# Process button
process_button_label = f"🔍 Process {len(selected_files)} File{'s' if len(selected_files) != 1 else ''}" if selected_files else "🔍 Process Files"
button_disabled = len(selected_files) == 0

if st.button(process_button_label, type="primary", disabled=button_disabled):
    with st.spinner("Processing prompts..."):
        # Create progress container
        progress_container = st.container()
        
        # Use selected files instead of all files
        prompt_files = selected_files
        total_files = len(prompt_files)
        
        if total_files == 0:
            st.warning("No files selected for processing.")
        else:
            # Process each file with progress tracking
            results = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, prompt_file in enumerate(prompt_files):
                status_text.text(f"Processing: {prompt_file}")
                
                file_path = os.path.join(prompts_folder, prompt_file)
                
                try:
                    # Read the prompt content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        prompt_content = f.read()
                    
                    # Get relevant context from the database
                    search_query = prompt_content[:500]
                    context = get_context(search_query, table, num_results=10)
                    
                    # Get LLM response
                    response = get_llm_response(prompt_content, context)
                    
                    results[prompt_file] = {
                        'status': 'Success',
                        'response': response
                    }
                except Exception as e:
                    results[prompt_file] = {
                        'status': 'Error',
                        'response': str(e)
                    }
                
                # Update progress
                progress_bar.progress((idx + 1) / total_files)
            
            status_text.text("Processing complete!")
            
            # Display results
            st.success(f"✅ Processed {len(results)} prompt files")
            
            # Extract scores and identify files not getting 10/10
            import re
            scores_summary = {}
            files_below_perfect = []
            
            for filename, result in results.items():
                if result['status'] == 'Success':
                    # Extract OVERALL SCORE using regex
                    score_pattern = r'OVERALL SCORE:\s*(\d+(?:\.\d+)?)/10'
                    match = re.search(score_pattern, result['response'], re.IGNORECASE)
                    
                    if match:
                        score = float(match.group(1))
                        scores_summary[filename] = score
                        if score < 10:
                            files_below_perfect.append((filename, score))
                    else:
                        scores_summary[filename] = "Score not found"
                        files_below_perfect.append((filename, "N/A"))
                else:
                    scores_summary[filename] = "Error"
            
            # Display summary of files not getting 10/10
            st.markdown("---")
            st.markdown("### 📊 Score Summary")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Total Files Processed", len(results))
                st.metric("Perfect Scores (10/10)", len(results) - len(files_below_perfect))
                st.metric("Below Perfect", len(files_below_perfect))
            
            with col2:
                if files_below_perfect:
                    st.warning("**Files NOT getting 10/10:**")
                    for filename, score in files_below_perfect:
                        score_display = f"{score}/10" if isinstance(score, (int, float)) else score
                        st.markdown(f"- **{filename.replace('.txt', '')}**: {score_display}")
                else:
                    st.success("🎉 All files received perfect scores (10/10)!")
            
            # Display all scores in an expandable section
            with st.expander("View All Scores"):
                scores_df = pd.DataFrame(
                    [(k.replace('.txt', ''), v) for k, v in scores_summary.items()],
                    columns=['Prompt File', 'Overall Score']
                )
                st.dataframe(scores_df, use_container_width=True)
            
            st.markdown("---")
            
            # Create tabs for each result
            tabs = st.tabs([name.replace('.txt', '') for name in results.keys()])
            
            for tab, (filename, result) in zip(tabs, results.items()):
                with tab:
                    if result['status'] == 'Success':
                        # Show score at the top if available
                        if filename in scores_summary and isinstance(scores_summary[filename], (int, float)):
                            score_color = "green" if scores_summary[filename] == 10 else "orange" if scores_summary[filename] >= 7 else "red"
                            st.markdown(f"<h3 style='color: {score_color};'>Score: {scores_summary[filename]}/10</h3>", unsafe_allow_html=True)
                        
                        st.markdown("### Evaluation Result")
                        st.markdown(result['response'])
                    else:
                        st.error(f"Error processing {filename}: {result['response']}")
            
            # Option to download results
            st.download_button(
                label="📥 Download All Results (Text)",
                data="\n\n".join([f"{'='*80}\n{filename}\n{'='*80}\n{r['response']}" 
                                 for filename, r in results.items()]),
                file_name="prompt_evaluation_results.txt",
                mime="text/plain"
            )

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    ### How it works:
    1. **Select** files to process (single/multiple/all)
    2. **Search** the document database for relevant context
    3. **Evaluate** using GPT-4 based on the prompt criteria
    4. **Display** results in organized tabs with scores
    
    ### Available Prompt Files:
    """)
    
    if os.path.exists(prompts_folder):
        for pf in available_files:
            # Show checkmark for selected files
            if pf in selected_files:
                st.markdown(f"✅ **{pf.replace('.txt', '')}**")
            else:
                st.markdown(f"◯ {pf.replace('.txt', '')}")
        
        st.markdown("---")
        st.metric("Total Available", len(available_files))
        st.metric("Selected", len(selected_files))
    else:
        st.warning("Prompts folder not found")
    
    st.markdown("---")
    st.markdown("### Settings")
    
    with st.expander("Advanced Options"):
        num_context_results = st.slider(
            "Number of context results per query",
            min_value=1,
            max_value=20,
            value=10,
            help="More results provide more context but may be slower"
        )
        
        temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lower values make output more deterministic"
        )