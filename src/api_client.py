import requests
import streamlit as st
from typing import Dict, Any, Optional
import logging


class PreprocessingAPI:
    def __init__(self, base_url: str = "http://localhost:8080"):
        print(f"Initializing PreprocessingAPI with base URL: {base_url}")
        self.base_url = base_url

    def health_check(self) -> bool:
        """Check if preprocessing service is available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def upload_dataset(
        self, file_content: bytes, filename: str
    ) -> Optional[Dict[str, Any]]:
        """Upload a dataset file"""
        try:
            files = {"file": (filename, file_content)}
            response = requests.post(f"{self.base_url}/upload", files=files, timeout=60)

            if response.status_code == 200:
                return response.json()
            else:
                st.error(
                    f"Upload failed: {response.json().get('error', 'Unknown error')}"
                )
                return None
        except Exception as e:
            st.error(f"Upload error: {str(e)}")
            return None

    def preprocess_dataset(
        self,
        dataset_source: str,
        dataset_id: str,
        sample_size: Optional[int] = None,
        format_config: Optional[Dict] = None,
        train_test_split: bool = False,
        test_size: float = 0.2,
    ) -> Optional[Dict[str, Any]]:
        """Start dataset preprocessing"""
        try:
            payload = {
                "dataset_source": dataset_source,
                "dataset_id": dataset_id,
                "options": {
                    "normalize_whitespace": True,
                    "train_test_split": train_test_split,
                    "test_size": test_size,
                },
            }

            if sample_size:
                payload["sample_size"] = sample_size

            if format_config:
                payload["options"]["format_config"] = format_config
            else:
                payload["options"]["format_config"] = {"type": "default"}

            response = requests.post(
                f"{self.base_url}/preprocess",
                json=payload,
                timeout=300,  # 5 minutes for processing
            )

            if response.status_code == 200:
                return response.json()
            else:
                st.error(
                    f"Preprocessing failed: {response.json().get('error', 'Unknown error')}"
                )
                return None
        except Exception as e:
            st.error(f"Preprocessing error: {str(e)}")
            return None

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a processed dataset"""
        try:
            response = requests.get(f"{self.base_url}/dataset/{dataset_id}", timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                st.error(
                    f"Failed to get dataset info: {response.json().get('error', 'Unknown error')}"
                )
                return None
        except Exception as e:
            st.error(f"Dataset info error: {str(e)}")
            return None


class TrainingAPI:
    def __init__(self, base_url: str = "http://localhost:8081"):
        print(f"Initializing TrainingAPI with base URL: {base_url}")
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if the training service is available"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Training service health check failed: {str(e)}")
            return False

    def start_training(
        self, processed_dataset_id: str, model_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Start training a model

        Args:
            processed_dataset_id: ID of the processed dataset from preprocessing service
            model_config: Configuration for the model and training parameters

        Returns:
            Training job response or None if failed
        """
        try:
            payload = {
                "processed_dataset_id": processed_dataset_id,
                "model_config": model_config,
            }

            response = self.session.post(
                f"{self.base_url}/train",
                json=payload,
                timeout=3600,  # 1 hour timeout for training
            )

            if response.status_code == 200:
                return response.json()
            else:
                logging.error(
                    f"Training failed: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logging.error(f"Training request failed: {str(e)}")
            return None

    def run_inference(self, job_id: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Run inference using a trained model"""
        try:
            payload = {"prompt": prompt}

            response = self.session.post(
                f"{self.base_url}/inference/{job_id}",
                json=payload,
                timeout=30,  # 30 seconds timeout for inference
            )

            if response.status_code == 200:
                return response.json()
            else:
                logging.error(
                    f"Inference failed: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logging.error(f"Inference request failed: {str(e)}")
            return None


training_api = TrainingAPI(
    base_url="https://training-service-18028528656.us-central1.run.app"
)
preprocessing_api = PreprocessingAPI(
    base_url="https://preprocessing-service-18028528656.us-central1.run.app"
)
