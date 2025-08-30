import openai
from openai import OpenAI
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class GPTFineTuner:
    def __init__(self, api_key: str):
        """Initialize GPT Fine-tuner"""
        self.client = OpenAI(api_key=api_key)
        self.fine_tuned_models = {}
    
    def prepare_training_data(self, df: pd.DataFrame, 
                            input_col: str = 'text', 
                            output_col: str = 'response',
                            system_message: str = None) -> str:
        """Prepare training data in the required format for fine-tuning"""
        
        if system_message is None:
            system_message = """Vous êtes un assistant IA spécialisé dans l'analyse des avis clients. 
            Votre rôle est de comprendre les préoccupations des clients, identifier les problèmes, 
            et fournir des réponses appropriées et empathiques."""
        
        training_data = []
        
        for _, row in df.iterrows():
            training_example = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": str(row[input_col])},
                    {"role": "assistant", "content": str(row[output_col])}
                ]
            }
            training_data.append(training_example)
        
        # Save to JSONL file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_data_{timestamp}.jsonl"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Prepared {len(training_data)} training examples in {filename}")
        return filename
    
    def upload_training_file(self, file_path: str) -> str:
        """Upload training file to OpenAI"""
        try:
            with open(file_path, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            file_id = response.id
            logger.info(f"Uploaded training file: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise
    
    def create_fine_tune_job(self, file_id: str, 
                           model: str = "gpt-4o-mini-2024-07-18",
                           hyperparameters: Dict = None) -> str:
        """Create a fine-tuning job"""
        
        if hyperparameters is None:
            hyperparameters = {
                "n_epochs": 3,
                "batch_size": 1,
                "learning_rate_multiplier": 0.1
            }
        
        try:
            response = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=model,
                hyperparameters=hyperparameters
            )
            
            job_id = response.id
            logger.info(f"Created fine-tuning job: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error creating fine-tune job: {e}")
            raise
    
    def monitor_fine_tune_job(self, job_id: str) -> Dict[str, Any]:
        """Monitor fine-tuning job status"""
        try:
            response = self.client.fine_tuning.jobs.retrieve(job_id)
            
            status_info = {
                'job_id': job_id,
                'status': response.status,
                'model': response.model,
                'fine_tuned_model': response.fine_tuned_model,
                'created_at': response.created_at,
                'finished_at': response.finished_at,
                'training_file': response.training_file,
                'result_files': response.result_files
            }
            
            logger.info(f"Job {job_id} status: {response.status}")
            return status_info
            
        except Exception as e:
            logger.error(f"Error monitoring job: {e}")
            raise
    
    def wait_for_completion(self, job_id: str, check_interval: int = 60) -> str:
        """Wait for fine-tuning job to complete and return model ID"""
        logger.info(f"Waiting for job {job_id} to complete...")
        
        while True:
            status_info = self.monitor_fine_tune_job(job_id)
            status = status_info['status']
            
            if status == 'succeeded':
                model_id = status_info['fine_tuned_model']
                logger.info(f"Fine-tuning completed! Model ID: {model_id}")
                return model_id
                
            elif status in ['failed', 'cancelled']:
                logger.error(f"Fine-tuning job {status}")
                raise Exception(f"Fine-tuning job {status}")
                
            else:
                logger.info(f"Job status: {status}. Checking again in {check_interval} seconds...")
                time.sleep(check_interval)
    
    def test_fine_tuned_model(self, model_id: str, test_messages: List[str]) -> List[str]:
        """Test the fine-tuned model with sample inputs"""
        results = []
        
        for message in test_messages:
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "user", "content": message}
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                
                result = response.choices[0].message.content
                results.append(result)
                logger.info(f"Test input: {message[:50]}...")
                logger.info(f"Model output: {result[:100]}...")
                
            except Exception as e:
                logger.error(f"Error testing model: {e}")
                results.append(f"Error: {str(e)}")
        
        return results
    
    def generate_synthetic_training_data(self, base_reviews: List[str], 
                                       num_examples: int = 100) -> pd.DataFrame:
        """Generate synthetic training data for fine-tuning"""
        
        synthetic_data = []
        
        # Define response templates based on sentiment and aspects
        response_templates = {
            'negative_service': "Je comprends votre frustration concernant le service. Nous prenons vos commentaires au sérieux et allons améliorer nos processus. Puis-je vous aider à résoudre ce problème?",
            'negative_product': "Je suis désolé que le produit n'ait pas répondu à vos attentes. Nous valorisons votre retour et souhaitons faire mieux. Pouvons-nous discuter d'une solution?",
            'negative_delivery': "Nous nous excusons pour les problèmes de livraison. C'est inacceptable et nous allons examiner cela avec notre équipe logistique. Comment pouvons-nous vous dédommager?",
            'positive': "Merci beaucoup pour ce retour positif! Nous sommes ravis que vous soyez satisfait. N'hésitez pas à nous contacter si vous avez besoin d'aide.",
            'neutral': "Merci pour votre commentaire. Nous apprécions votre retour et restons à votre disposition pour toute question."
        }
        
        # Generate examples (simplified - in practice, you'd use more sophisticated generation)
        for i in range(num_examples):
            # This would be more sophisticated in practice
            review = f"Exemple d'avis client {i+1}"
            response_type = list(response_templates.keys())[i % len(response_templates)]
            response = response_templates[response_type]
            
            synthetic_data.append({
                'text': review,
                'response': response,
                'category': response_type
            })
        
        return pd.DataFrame(synthetic_data)
    
    def save_model_info(self, model_id: str, job_id: str, 
                       metadata: Dict[str, Any], file_path: Path):
        """Save fine-tuned model information"""
        model_info = {
            'model_id': model_id,
            'job_id': job_id,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        with open(file_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model info saved to {file_path}")

# Example usage
if __name__ == "__main__":
    from src.utils.config import settings
    
    # Initialize fine-tuner
    fine_tuner = GPTFineTuner(settings.OPENAI_API_KEY)
    
    # Generate synthetic training data
    synthetic_df = fine_tuner.generate_synthetic_training_data([], num_examples=50)
    
    # Prepare training data
    training_file = fine_tuner.prepare_training_data(synthetic_df)
    
    print(f"Training data prepared: {training_file}")
    print(f"Number of examples: {len(synthetic_df)}")
