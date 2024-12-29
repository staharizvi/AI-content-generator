from typing import List, Optional, Dict
from dataclasses import dataclass
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from pathlib import Path
import json
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration settings for text generation."""
    max_length: int = 512
    min_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1
    do_sample: bool = True
    repetition_penalty: float = 1.2

class ContentType:
    """Enum-like class for content types and their prompts."""
    BLOG_POST = "blog_post"
    PRODUCT_DESCRIPTION = "product_description"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"

    PROMPTS = {
        BLOG_POST: """<s>[INST] Write a {sentiment} blog post about {topic} that focuses on {aspect}. 
        Make it engaging and informative while maintaining a {sentiment} tone throughout the content. [/INST]""",
        
        PRODUCT_DESCRIPTION: """<s>[INST] Create a {sentiment} product description for {product} highlighting its {features}. 
        The description should be compelling and focus on the benefits to the customer. [/INST]""",
        
        SOCIAL_MEDIA: """<s>[INST] Write a {sentiment} social media post about {topic} for {platform}. 
        Make it engaging and appropriate for the platform's style while maintaining a {sentiment} tone. [/INST]""",
        
        EMAIL: """<s>[INST] Write a {sentiment} email about {topic} for {purpose}. 
        Ensure it's professional and achieves its purpose while maintaining a {sentiment} tone. [/INST]"""
    }

    @classmethod
    def get_content_types(cls):
        return [cls.BLOG_POST, cls.PRODUCT_DESCRIPTION, cls.SOCIAL_MEDIA, cls.EMAIL]

class ModelManager:
    """Handles model loading and management."""
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1", token: str = None):
        self.model_name = model_name
        self.token = token
        logger.info("Initializing model manager...")
        
        try:
            # Load tokenizer with authentication
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=self.token,
                trust_remote_code=True
            )
            
            # Load model with optimizations and authentication
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=self.token,
                torch_dtype=torch.float16,
                device_map="auto",  # Let accelerate handle device management
                load_in_8bit=True,
                trust_remote_code=True
            )
            
            # Create generator pipeline without specifying device
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer
            )
            
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def get_generator(self):
        """Returns the text generation pipeline."""
        return self.generator
    
class ContentGenerator:
    """Main class for generating AI content."""
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.generator = model_manager.get_generator()

    def _format_prompt(self, content_type: str, parameters: Dict[str, str]) -> str:
        """Formats the prompt template with given parameters."""
        if content_type not in ContentType.PROMPTS:
            raise ValueError(f"Invalid content type. Choose from: {list(ContentType.PROMPTS.keys())}")
        
        try:
            prompt = ContentType.PROMPTS[content_type].format(**parameters)
            return prompt
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")

    def _clean_generated_text(self, text: str, prompt: str) -> str:
        """Cleans and formats the generated text."""
        # Remove the prompt from the generated text
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()

    def generate_content(
        self,
        content_type: str,
        parameters: Dict[str, str],
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """Generates content based on specified type and parameters."""
        if config is None:
            config = GenerationConfig()

        try:
            prompt = self._format_prompt(content_type, parameters)
            logger.info(f"Generated prompt: {prompt}")

            outputs = self.generator(
                prompt,
                max_length=config.max_length,
                min_length=config.min_length,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                num_return_sequences=config.num_return_sequences,
                do_sample=config.do_sample
            )

            # Clean and format the generated texts
            generated_texts = [
                self._clean_generated_text(output['generated_text'], prompt)
                for output in outputs
            ]

            return generated_texts

        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise

class ContentManager:
    """Manages content generation and saves results."""
    def __init__(self, output_dir: str = "generated_content"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_content(self, content: List[str], metadata: Dict):
        """Saves generated content with metadata."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{metadata['content_type']}_{timestamp}.json"
            output_path = self.output_dir / filename
            
            output_data = {
                "metadata": metadata,
                "content": content
            }
            
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Content saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving content: {str(e)}")
            raise

def get_user_input() -> Dict[str, str]:
    """Gets content generation parameters from user."""
    print("\n=== AI Content Generator using Mistral-7B ===")
    
    # Show available content types
    print("\nAvailable content types:")
    for i, content_type in enumerate(ContentType.get_content_types(), 1):
        print(f"{i}. {content_type}")
    
    while True:
        try:
            choice = int(input("\nSelect content type (1-4): ")) - 1
            if 0 <= choice < len(ContentType.get_content_types()):
                content_type = ContentType.get_content_types()[choice]
                break
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")

    # Get common parameters
    topic = input("\nEnter the main topic: ")
    sentiment = input("Enter the desired sentiment (e.g., professional, casual, enthusiastic): ")
    
    # Get content-specific parameters
    parameters = {
        "topic": topic,
        "sentiment": sentiment
    }
    
    if content_type == ContentType.BLOG_POST:
        parameters["aspect"] = input("Enter the specific aspect to focus on: ")
    elif content_type == ContentType.PRODUCT_DESCRIPTION:
        parameters["product"] = topic  # Use topic as product
        parameters["features"] = input("Enter the key features to highlight: ")
    elif content_type == ContentType.SOCIAL_MEDIA:
        parameters["platform"] = input("Enter the social media platform: ")
    elif content_type == ContentType.EMAIL:
        parameters["purpose"] = input("Enter the email purpose: ")
    
    return {
        "content_type": content_type,
        "parameters": parameters
    }

def main():
    try:
        # Your Hugging Face token
        HF_TOKEN = "SECRET ACCESS KEY"
        if not HF_TOKEN:
            raise ValueError("Please set the HUGGING_FACE_TOKEN environment variable")

        # Initialize components
        logger.info("Initializing the model (this may take a few minutes)...")
        model_manager = ModelManager(token=HF_TOKEN)
        content_generator = ContentGenerator(model_manager)
        content_manager = ContentManager()

        # Get user input
        user_input = get_user_input()
        
        # Configure generation parameters
        config = GenerationConfig(
            max_length=750,
            temperature=0.7,
            repetition_penalty=1.2
        )
        
        # Generate content
        generated_content = content_generator.generate_content(
            content_type=user_input["content_type"],
            parameters=user_input["parameters"],
            config=config
        )
        
        # Save the generated content with metadata
        metadata = {
            "content_type": user_input["content_type"],
            "parameters": user_input["parameters"],
            "generation_config": vars(config),
            "model": "mistralai/Mistral-7B-Instruct-v0.1"
        }
        
        output_path = content_manager.save_content(
            content=generated_content,
            metadata=metadata
        )
        
        # Print the generated content
        print("\nGenerated Content:")
        for i, text in enumerate(generated_content, 1):
            print(f"\nVersion {i}:\n{text}\n")
        print(f"\nContent saved to: {output_path}")

    except ValueError as ve:
        logger.error(f"Configuration error: {str(ve)}")
        print(f"\nConfiguration error: {str(ve)}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"\nAn unexpected error occurred: {str(e)}")
        print("Please ensure you have sufficient memory and all requirements installed.")

if __name__ == "__main__":
    main()
