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
            
            # Extract compliance status from results
            import re
            compliance_summary = {}
            non_compliant_files = []
            compliant_files = []
            
            for filename, result in results.items():
                if result['status'] == 'Success':
                    # Extract Overall Compliance using regex
                    # Try multiple patterns to catch different formats
                    patterns = [
                        r'Overall\s*Compliance[:\s]*\*?\*?\s*(True|False)\b',
                        r'Overall\s+Compliance[:\s]+(True|False)',
                        r'Overall\s+Compliance:\s*(True|False)',
                        r'Overall Compliance:\s*(True|False)',
                        r'Overall Compliance\s+(True|False)'
                    ]
                    
                    # Save response to debug file for inspection
                    debug_filename = f"debug_{filename.replace('.txt', '')}_response.txt"
                    with open(debug_filename, 'w', encoding='utf-8') as debug_file:
                        debug_file.write(f"Response for {filename}:\n")
                        debug_file.write("="*80 + "\n")
                        debug_file.write(result['response'])
                        debug_file.write("\n" + "="*80 + "\n")
                    
                    match = None
                    for pattern in patterns:
                        match = re.search(pattern, result['response'], re.IGNORECASE | re.MULTILINE)
                        if match:
                            break
                    
                    if match:
                        compliance_status = match.group(1).lower() == 'true'                    
                        compliance_summary[filename] = compliance_status
                        if compliance_status:
                            compliant_files.append(filename)
                        else:
                            non_compliant_files.append(filename)
                    else:
                        # Also show in Streamlit what text was searched
                        st.warning(f"Could not find 'Overall Compliance' in {filename}")
                        with st.expander(f"Show last 500 characters of {filename} response"):
                            st.text(result['response'][-500:])
                        compliance_summary[filename] = "Not found"
                        non_compliant_files.append(filename)
                else:
                    compliance_summary[filename] = "Error"
                    non_compliant_files.append(filename)
            
            # Display compliance summary
            st.markdown("---")
            st.markdown("### 📊 Compliance Summary")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Total Files Processed", len(results))
                st.metric("✅ Compliant", len(compliant_files))
                st.metric("❌ Non-Compliant", len(non_compliant_files))
            
            with col2:
                if non_compliant_files:
                    st.error("**Files NOT Compliant:**")
                    for filename in non_compliant_files:
                        status = compliance_summary.get(filename, "Unknown")
                        if status == "Error":
                            status_display = "⚠️ Error"
                        elif status == "Not found":
                            status_display = "❓ Compliance status not found"
                        else:
                            status_display = "❌ False"
                        st.markdown(f"- **{filename.replace('.txt', '')}**: {status_display}")
                
                if compliant_files:
                    st.success("**Compliant Files:**")
                    for filename in compliant_files:
                        st.markdown(f"- ✅ **{filename.replace('.txt', '')}**")
            
            # Display all compliance results in an expandable section
            with st.expander("View All Compliance Results"):
                compliance_data = []
                for k, v in compliance_summary.items():
                    if v == True:
                        status_str = "✅ Compliant"
                    elif v == False:
                        status_str = "❌ Non-Compliant"
                    elif v == "Error":
                        status_str = "⚠️ Error"
                    elif v == "Not found":
                        status_str = "❓ Not Found"
                    else:
                        status_str = str(v)
                    compliance_data.append((k.replace('.txt', ''), status_str))
                
                compliance_df = pd.DataFrame(
                    compliance_data,
                    columns=['Prompt File', 'Compliance Status']
                )
                st.dataframe(compliance_df, use_container_width=True)
            
            st.markdown("---")
            
            # Create tabs for each result
            tabs = st.tabs([name.replace('.txt', '') for name in results.keys()])
            
            for tab, (filename, result) in zip(tabs, results.items()):
                with tab:
                    if result['status'] == 'Success':
                        # Show compliance status at the top if available
                        if filename in compliance_summary:
                            compliance_status = compliance_summary[filename]
                            if compliance_status == True:
                                st.markdown("<h3 style='color: green;'>✅ Compliant</h3>", unsafe_allow_html=True)
                            elif compliance_status == False:
                                st.markdown("<h3 style='color: red;'>❌ Non-Compliant</h3>", unsafe_allow_html=True)
                            elif compliance_status == "Not found":
                                st.markdown("<h3 style='color: orange;'>❓ Compliance Status Not Found</h3>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<h3 style='color: gray;'>Status: {compliance_status}</h3>", unsafe_allow_html=True)
                        
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