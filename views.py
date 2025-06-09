import streamlit as st
import time
import re
from api_client import preprocessing_api, training_api


def show_sidebar():
    st.title("🧠 Gemma Fine-Tuning")
    st.markdown("---")
    st.subheader("Progress")

    # Progress for Dataset Selection
    st.markdown("1️⃣ **Dataset Selection**")
    if st.session_state.get("step_1_completed", False):
        st.success("✓ Completed")
    else:
        st.info(
            "◯ In progress" if st.session_state.current_step == 1 else "◯ Not started"
        )

    # Progress for Data Processing
    st.markdown("2️⃣ **Data Processing**")
    if st.session_state.get("step_2_completed", False):
        st.success("✓ Completed")
    elif st.session_state.get("step_1_completed", False):
        st.info(
            "◯ In progress" if st.session_state.current_step == 2 else "◯ Not started"
        )
    else:
        st.info("◯ Not started")

    # Progress for Model Configuration
    st.markdown("3️⃣ **Model Configuration**")
    if st.session_state.get("step_3_completed", False):
        st.success("✓ Completed")
    elif st.session_state.get("step_2_completed", False):
        st.info(
            "◯ In progress" if st.session_state.current_step == 3 else "◯ Not started"
        )
    else:
        st.info("◯ Not started")

    # Progress for Training & Results
    st.markdown("4️⃣ **Training & Results**")
    if st.session_state.get("step_4_completed", False):
        st.success("✓ Completed")
    elif st.session_state.get("step_3_completed", False):
        st.info(
            "◯ In progress" if st.session_state.current_step == 4 else "◯ Not started"
        )
    else:
        st.info("◯ Not started")
    st.markdown("---")

    # Navigation buttons
    if st.button("Dataset Selection", use_container_width=True):
        st.session_state.current_step = 1
        st.rerun()

    if st.button(
        "Data Processing",
        use_container_width=True,
        disabled=not st.session_state.get("step_1_completed", False),
    ):
        st.session_state.current_step = 2
        st.rerun()

    if st.button(
        "Model Configuration",
        use_container_width=True,
        disabled=not st.session_state.get("step_2_completed", False),
    ):
        st.session_state.current_step = 3
        st.rerun()

    if st.button(
        "Training & Results",
        use_container_width=True,
        disabled=not st.session_state.get("step_3_completed", False),
    ):
        st.session_state.current_step = 4
        st.rerun()


def show_current_view():
    if st.session_state.current_step == 1:
        show_dataset_selection()
    elif st.session_state.current_step == 2:
        show_data_processing()
    elif st.session_state.current_step == 3:
        show_model_configuration()
    elif st.session_state.current_step == 4:
        show_training_and_results()
    elif st.session_state.current_step == 5:
        show_inference_testing()


def show_dataset_selection():
    st.title("📚 Dataset Selection")

    # Check preprocessing service status
    if not preprocessing_api.health_check():
        st.error(
            "⚠️ Preprocessing service is not available. Please start the preprocessing service on localhost:8080"
        )
        st.code("cd services/preprocessing && python app.py")
        return
    else:
        st.success("✅ Preprocessing service is available")

    tab1, tab2, tab3 = st.tabs(
        ["Upload Dataset", "Standard Datasets", "Hugging Face Dataset"]
    )

    with tab1:
        st.header("Upload Your Dataset")

        uploaded_file = st.file_uploader(
            "Choose a file", type=["csv", "json", "jsonl", "txt"]
        )

        if uploaded_file is not None:
            if st.button(
                "Upload Dataset", key="upload_dataset", use_container_width=True
            ):
                with st.spinner("Uploading dataset..."):
                    file_content = uploaded_file.read()
                    result = preprocessing_api.upload_dataset(
                        file_content, uploaded_file.name
                    )

                    if result:
                        st.success("✅ Dataset uploaded successfully!")
                        st.json(result)

                        # Store upload info in session state
                        st.session_state.uploaded_dataset = {
                            "dataset_id": result["dataset_id"],
                            "filename": result["filename"],
                            "size_bytes": result["size_bytes"],
                        }
                        st.session_state.dataset_source = "upload"
                        st.session_state.step_1_completed = True
                        st.session_state.current_step = 2
                        st.rerun()

    with tab2:
        st.header("Standard Datasets")

        # Use actual Hugging Face dataset names
        dataset_options = {
            "philschmid/gretel-synthetic-text-to-sql": "Gretel Synthetic Text-to-SQL",
            # NOTE: The following might not be supported yet in the backend lol
            "squad_v2": "SQuAD v2.0 - Question Answering",
            "xsum": "XSum - Summarization",
            "OpenAssistant/oasst1": "Open Assistant - Instruction Following",
        }

        selected_dataset = st.selectbox(
            "Select a standard dataset:",
            list(dataset_options.keys()),
            format_func=lambda x: dataset_options[x],
        )

        # Dataset descriptions
        if selected_dataset == "philschmid/gretel-synthetic-text-to-sql":
            st.info(
                "Synthetic text-to-SQL dataset with 100k+ examples for fine-tuning SQL generation models"
            )
        elif selected_dataset == "squad_v2":
            st.info(
                "100,000+ question-answer pairs on 500+ articles with unanswerable questions"
            )
        elif selected_dataset == "xsum":
            st.info("BBC articles with single sentence summaries")
        elif selected_dataset == "OpenAssistant/oasst1":
            st.info("Open-source instruction following dataset with human feedback")

        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.number_input(
                "Sample Size", min_value=100, max_value=50000, value=1000, step=100
            )
        with col2:
            shuffle_dataset = st.checkbox("Shuffle Dataset", value=True)

        if st.button("Use This Dataset", key="use_standard", use_container_width=True):
            with st.spinner("Selecting dataset..."):
                # Store dataset info in session state
                st.session_state.standard_dataset = {
                    "dataset_id": selected_dataset,
                    "dataset_name": dataset_options[selected_dataset],
                    "sample_size": sample_size,
                    "shuffle": shuffle_dataset,
                }
                st.session_state.dataset_source = "standard"
                st.session_state.step_1_completed = True
                st.session_state.current_step = 2
                st.success("✅ Dataset selected!")
                time.sleep(1)
                st.rerun()

    with tab3:
        st.header("Hugging Face Dataset")

        st.write(
            "Enter any Hugging Face dataset ID that's compatible with `datasets.load_dataset()`"
        )

        # Dataset ID input
        hf_dataset_id = st.text_input(
            "Hugging Face Dataset ID",
            placeholder="e.g., squad, microsoft/DialoGPT-medium, databricks/databricks-dolly-15k",
            help="Full dataset path on Hugging Face Hub",
        )

        # Optional subset/split configuration
        col1, col2 = st.columns(2)
        with col1:
            dataset_subset = st.text_input(
                "Subset (optional)",
                placeholder="e.g., plain_text",
                help="Dataset subset if applicable",
            )
        with col2:
            dataset_split = st.text_input(
                "Split (optional)",
                value="train",
                help="Dataset split to use (train, test, validation)",
            )

        # Sample size
        col1, col2 = st.columns(2)
        with col1:
            sample_size_hf = st.number_input(
                "Sample Size",
                min_value=100,
                max_value=50000,
                value=1000,
                step=100,
                key="hf_sample_size",
            )
        with col2:
            shuffle_dataset_hf = st.checkbox(
                "Shuffle Dataset", value=True, key="hf_shuffle"
            )

        # Examples section
        with st.expander("💡 Popular Dataset Examples"):
            examples = {
                "squad": "Stanford Question Answering Dataset",
                "microsoft/DialoGPT-medium": "Conversational AI dataset",
                "databricks/databricks-dolly-15k": "Instruction following dataset",
                "tatsu-lab/alpaca": "Alpaca instruction dataset",
                "openai/summarize_from_feedback": "Summarization with human feedback",
                "timdettmers/openassistant-guanaco": "OpenAssistant Guanaco dataset",
            }

            for dataset_id, description in examples.items():
                if st.button(
                    f"📋 {dataset_id}", help=description, key=f"example_{dataset_id}"
                ):
                    st.session_state.hf_dataset_example = dataset_id
                    st.rerun()

        # Auto-fill from examples
        if st.session_state.get("hf_dataset_example"):
            hf_dataset_id = st.session_state.hf_dataset_example
            del st.session_state.hf_dataset_example
            st.rerun()

        if hf_dataset_id and st.button(
            "Use This Dataset", key="use_hf", use_container_width=True
        ):
            with st.spinner("Validating Hugging Face dataset..."):
                # Store dataset info in session state
                st.session_state.hf_dataset = {
                    "dataset_id": hf_dataset_id,
                    "subset": dataset_subset if dataset_subset else None,
                    "split": dataset_split if dataset_split else "train",
                    "sample_size": sample_size_hf,
                    "shuffle": shuffle_dataset_hf,
                }
                st.session_state.dataset_source = "huggingface"
                st.session_state.step_1_completed = True
                st.session_state.current_step = 2
                st.success("✅ Hugging Face dataset selected!")
                time.sleep(1)
                st.rerun()


def show_data_processing():
    st.title("🔄 Data Processing")

    # Show selected dataset info
    if st.session_state.get("dataset_source") == "upload":
        dataset_info = st.session_state.get("uploaded_dataset", {})
        st.info(
            f"📁 Selected Dataset: {dataset_info.get('filename', 'Uploaded file')} ({dataset_info.get('size_bytes', 0)} bytes)"
        )
    elif st.session_state.get("dataset_source") == "standard":
        dataset_info = st.session_state.get("standard_dataset", {})
        st.info(
            f"🤗 Selected Dataset: {dataset_info.get('dataset_name', 'Standard dataset')} (Sample size: {dataset_info.get('sample_size', 'N/A')})"
        )
    elif st.session_state.get("dataset_source") == "huggingface":
        dataset_info = st.session_state.get("hf_dataset", {})
        subset_info = (
            f"/{dataset_info.get('subset')}" if dataset_info.get("subset") else ""
        )
        split_info = (
            f" [{dataset_info.get('split', 'train')}]"
            if dataset_info.get("split")
            else ""
        )
        st.info(
            f"🤗 Selected Dataset: {dataset_info.get('dataset_id', 'HF dataset')}{subset_info}{split_info} (Sample size: {dataset_info.get('sample_size', 'N/A')})"
        )
    else:
        st.error("No dataset selected. Please go back to dataset selection.")
        return

    # Format Configuration
    st.header("Format Configuration")

    format_type = st.selectbox(
        "Format Type",
        ["default", "custom"],
        format_func=lambda x: {
            "default": "Default (Keep original format)",
            "custom": "Custom Conversation Format",
        }[x],
    )

    format_config = {"type": format_type}

    if format_type == "default":
        # Show description for default format
        st.info("""
        📋 **Default Format Requirements**
        
        Your dataset should already be properly formatted for training. Expected format:
        
        ```json
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "User's question or input"},
                {"role": "assistant", "content": "Assistant's response"}
            ]
        }
        ```
        
        **Alternative formats:**
        - `{"input": "...", "output": "..."}` 
        - `{"question": "...", "answer": "..."}`
        - `{"instruction": "...", "response": "..."}`
        
        If your dataset doesn't match these formats, please use "Custom Conversation Format" instead.
        """)

    elif format_type == "custom":
        # System and User prompt templates side by side
        col1, col2 = st.columns(2)

        with col1:
            include_system = st.checkbox("Include System Message", value=False)
            if include_system:
                system_message = st.text_area(
                    "System Message",
                    value="You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.",
                    help="Define the role and behavior of the assistant",
                    height=200,
                )
                format_config["system_message"] = system_message
                format_config["include_system"] = True

        with col2:
            # User prompt template
            default_template = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""

            user_prompt_template = st.text_area(
                "User Prompt Template",
                value=default_template,
                help="Template for user prompts. Use {field_name} placeholders (e.g., {question}, {context}) to reference dataset fields",
                height=200,
            )
            format_config["user_prompt_template"] = user_prompt_template

        # Extract placeholders from user prompt template
        placeholders = re.findall(r"\{(\w+)\}", user_prompt_template)

        if placeholders:
            st.subheader("Field Mapping")
            st.write("Map the placeholders in your template to fields in your dataset:")

            # Create a dynamic table for field mapping
            field_mapping = {}

            # Create columns for the mapping table
            col_headers = st.columns([2, 3, 3])
            with col_headers[0]:
                st.write("**Placeholder**")
            with col_headers[1]:
                st.write("**Dataset Field**")
            with col_headers[2]:
                st.write("**Description**")

            # Create rows for each placeholder
            for placeholder in set(placeholders):  # Remove duplicates
                col_values = st.columns([2, 3, 3])

                with col_values[0]:
                    st.code(f"{{{placeholder}}}")

                with col_values[1]:
                    # Provide smart defaults based on placeholder name
                    if placeholder.lower() in ["question", "query", "input"]:
                        default_field = "sql_prompt"
                    elif placeholder.lower() in ["context", "schema", "background"]:
                        default_field = "sql_context"
                    elif placeholder.lower() in ["answer", "output", "response"]:
                        default_field = "sql"
                    else:
                        default_field = placeholder

                    field_mapping[placeholder] = st.text_input(
                        f"Field for {placeholder}",
                        value=default_field,
                        key=f"field_{placeholder}",
                        label_visibility="collapsed",
                    )

                with col_values[2]:
                    descriptions = {
                        "question": "The user question or query",
                        "context": "Additional context or schema information",
                        "query": "The user question or query",
                        "schema": "Database schema information",
                        "input": "The input text or question",
                        "background": "Background context information",
                    }
                    st.write(descriptions.get(placeholder.lower(), "Custom field"))

            # Update format config with field mappings
            format_config.update(
                {
                    "input_field": field_mapping.get(
                        "question",
                        field_mapping.get("query", field_mapping.get("input", "input")),
                    ),
                    "output_field": "sql",  # Default output field
                    "context_field": field_mapping.get(
                        "context", field_mapping.get("schema", "context")
                    ),
                }
            )

            # Store all field mappings for the preprocessing service
            format_config["field_mappings"] = field_mapping
        else:
            st.info(
                "Add placeholders like {question} or {context} to your template to configure field mapping."
            )

        # Quick templates
        st.subheader("Quick Templates")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("SQL Template", help="Load SQL generation template"):
                st.session_state.template_reload = "sql"
                st.rerun()

        with col2:
            if st.button("QA Template", help="Load Question-Answer template"):
                st.session_state.template_reload = "qa"
                st.rerun()

        with col3:
            if st.button(
                "Instruction Template", help="Load Instruction-following template"
            ):
                st.session_state.template_reload = "instruction"
                st.rerun()

        # Handle template reloading
        if st.session_state.get("template_reload"):
            template_type = st.session_state.template_reload
            del st.session_state.template_reload

            if template_type == "sql":
                st.info("SQL template loaded! Update the fields above as needed.")
            elif template_type == "qa":
                st.info(
                    "Set Input Field: 'question', Output Field: 'answer', Template: 'Question: {question}\\nAnswer:'"
                )
            elif template_type == "instruction":
                st.info(
                    "Set Input Field: 'instruction', Output Field: 'response', Template: 'Instruction: {instruction}\\nResponse:'"
                )

    # Dataset Splitting
    st.header("Dataset Splitting")

    train_test_split = st.checkbox("Split into Train/Test Sets", value=True)
    test_size = 0.2

    if train_test_split:
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)

    # Process Data Button
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Dataset Selection", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()

    with col2:
        if st.button("🔄 Process Dataset", use_container_width=True):
            with st.spinner("Processing dataset..."):
                # Get dataset info
                if st.session_state.dataset_source == "upload":
                    dataset_id = st.session_state.uploaded_dataset["dataset_id"]
                    sample_size = None
                elif st.session_state.dataset_source == "standard":
                    dataset_id = st.session_state.standard_dataset["dataset_id"]
                    sample_size = st.session_state.standard_dataset["sample_size"]
                elif st.session_state.dataset_source == "huggingface":
                    dataset_id = st.session_state.hf_dataset["dataset_id"]
                    sample_size = st.session_state.hf_dataset["sample_size"]

                # Call preprocessing API
                result = preprocessing_api.preprocess_dataset(
                    dataset_source=st.session_state.dataset_source,
                    dataset_id=dataset_id,
                    sample_size=sample_size,
                    format_config=format_config,
                    train_test_split=train_test_split,
                    test_size=test_size,
                )

                if result:
                    # Store processing results
                    st.session_state.processing_results = result
                    st.session_state.step_2_completed = True
                    st.rerun()

    # Show processing results if available
    if st.session_state.get("processing_results"):
        result = st.session_state.processing_results

        st.success("✅ Data processing completed!")

        # Show results in full width
        st.subheader("📊 Processing Results")

        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Original Samples", result["original_count"])
        with col2:
            if train_test_split and "train_count" in result:
                st.metric("Train Samples", result["train_count"])
            else:
                st.metric("Processed Samples", result.get("processed_count", 0))
        with col3:
            if train_test_split and "test_count" in result:
                st.metric("Test Samples", result["test_count"])
        with col4:
            st.metric("Dataset ID", result["processed_dataset_id"][:8] + "...")

        # Show sample comparison in full width
        if "sample_comparison" in result:
            st.subheader("🔍 Sample Comparison")

            # Original sample
            st.markdown("**Original Sample:**")
            st.json(result["sample_comparison"]["original"])

            # Processed sample
            st.markdown("**Processed Sample:**")
            st.json(result["sample_comparison"]["processed"])

        # Navigation buttons at the bottom
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Dataset Selection", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()

        with col2:
            if st.button(
                "Continue to Model Configuration →",
                use_container_width=True,
            ):
                st.session_state.current_step = 3
                st.rerun()

    # If no processing results yet, show back button only
    elif not st.session_state.get("processing_results"):
        st.markdown("---")
        if st.button("← Back to Dataset Selection", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()


def show_model_configuration():
    st.title("⚙️ Model Configuration")

    # Check training service status
    if not training_api.health_check():
        st.error(
            "⚠️ Training service is not available. Please start the training service on localhost:8081"
        )
        st.code("cd services/training && python app.py")
        return
    else:
        st.success("✅ Training service is available")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Selection")

        model_id = st.selectbox(
            "Base Model",
            ["google/gemma-3-1b-pt", "google/gemma-3-4b-pt"],
            format_func=lambda x: {
                "google/gemma-3-1b-pt": "Gemma 3 1B (Fastest)",
                "google/gemma-3-4b-pt": "Gemma 3 4B (Better Quality)",
            }[x],
        )

        method = st.selectbox(
            "Fine-tuning Method",
            ["LoRA", "QLoRA"],
            format_func=lambda x: {
                "LoRA": "LoRA (Low-Rank Adaptation)",
                "QLoRA": "QLoRA (Quantized LoRA - Less Memory)",
            }[x],
        )

    with col2:
        st.subheader("Hyperparameters")

        epochs = st.slider("Epochs", 1, 10, 3)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
            value=2e-4,
            format_func=lambda x: f"{x:.0e}",
        )

        lora_rank = st.slider("LoRA Rank", 4, 64, 16, step=4)
        lora_alpha = st.slider("LoRA Alpha", 8, 128, 32, step=8)

    # Save configuration
    model_config = {
        "model_id": model_id,
        "method": method,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "batch_size": 1,
        "max_seq_length": 512,
        "gradient_accumulation_steps": 4,
    }

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Data Processing", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()

    with col2:
        if st.button("💾 Save Configuration", use_container_width=True):
            st.session_state.model_config = model_config
            st.session_state.step_3_completed = True
            st.session_state.current_step = 4
            st.success("✅ Model configuration saved!")
            time.sleep(1)
            st.rerun()


def show_training_and_results():
    st.title("🚀 Training & Results")

    # Check prerequisites
    if not st.session_state.get("processing_results"):
        st.error(
            "❌ No processed dataset found. Please complete data processing first."
        )
        return

    if not st.session_state.get("model_config"):
        st.error(
            "❌ No model configuration found. Please complete model configuration first."
        )
        return

    processed_dataset_id = st.session_state.processing_results["processed_dataset_id"]

    # Show configuration summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset ID", processed_dataset_id[:12] + "...")
        st.metric("Model", st.session_state.model_config["model_id"].split("/")[-1])
        st.metric("Method", st.session_state.model_config["method"])

    with col2:
        st.metric("Epochs", st.session_state.model_config["epochs"])
        st.metric(
            "Learning Rate", f"{st.session_state.model_config['learning_rate']:.0e}"
        )
        st.metric("LoRA Rank", st.session_state.model_config["lora_rank"])

    st.markdown("---")

    # Training controls
    if not st.session_state.get("training_job_id"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Model Config", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()

        with col2:
            if st.button("🚀 Start Training", use_container_width=True):
                with st.spinner(
                    "Training in progress... This may take several minutes"
                ):
                    # Start training
                    result = training_api.start_training(
                        processed_dataset_id=processed_dataset_id,
                        model_config=st.session_state.model_config,
                    )

                    if result:
                        st.session_state.training_result = result
                        st.session_state.training_job_id = result.get("job_id")
                        st.session_state.step_4_completed = True
                        st.rerun()
                    else:
                        st.error("❌ Failed to start training")

    # Show training result if available
    if st.session_state.get("training_job_id"):
        job_id = st.session_state.training_job_id

        st.success("✅ Training completed!")
        st.subheader("🎯 Training Results")

        # Show job info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Job ID", job_id[:12] + "...")

        # Show adapter path if available
        if st.session_state.get("training_result"):
            result = st.session_state.training_result
            adapter_path = result.get("result", {}).get("adapter_path")
            if adapter_path:
                st.info(f"💾 **Trained Adapter:** `{adapter_path}`")
                st.info("💡 The LoRA adapter has been saved to Google Cloud Storage.")

        # Next steps
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start New Training", use_container_width=True):
                st.session_state.training_job_id = None
                st.session_state.training_result = None
                st.rerun()

        with col2:
            if st.button("🔬 Test Model", use_container_width=True):
                st.session_state.current_step = 5
                st.rerun()


def show_inference_testing():
    st.title("🔬 Model Testing")

    # Check if we have a trained model
    if not st.session_state.get("training_job_id"):
        st.error("❌ No trained model found. Please complete training first.")

        if st.button("← Back to Training", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()
        return

    # Show model info
    job_id = st.session_state.training_job_id
    st.info(f"🤖 Testing model from job: {job_id}")

    # Input form
    with st.form("inference_form"):
        prompt = st.text_area(
            "Enter your prompt",
            height=300,
            help="Enter the prompt for the model",
            placeholder="""Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
CREATE TABLE salesperson (salesperson_id INT, name TEXT, region TEXT); INSERT INTO salesperson (salesperson_id, name, region) VALUES (1, 'John Doe', 'North'), (2, 'Jane Smith', 'South'); CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL, sale_date DATE); INSERT INTO timber_sales (sales_id, salesperson_id, volume, sale_date) VALUES (1, 1, 120, '2021-01-01'), (2, 1, 150, '2021-02-01'), (3, 2, 180, '2021-01-01');
</SCHEMA>

<USER_QUERY>
What is the total volume of timber sold by each salesperson, sorted by salesperson?
</USER_QUERY>""",
        )

        submitted = st.form_submit_button("🚀 Generate", use_container_width=True)

    if submitted:
        if not prompt:
            st.error("Please enter a prompt")
            return

        with st.spinner("Generating response..."):
            try:
                response = training_api.run_inference(job_id=job_id, prompt=prompt)

                if response:
                    st.success("✅ Generation successful!")

                    # Show results
                    st.subheader("Generated Response")
                    st.markdown(response["result"])

                    # Add "Try Another" button
                    if st.button("🔄 Try Another Prompt", use_container_width=True):
                        st.rerun()

            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")

    # Navigation
    st.markdown("---")
    if st.button("← Back to Training Results", use_container_width=True):
        st.session_state.current_step = 4
        st.rerun()
