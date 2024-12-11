import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GemmaPromptTester:
    BASE_PROMPT = """You are EmotiChat, an advanced AI assistant specialized in emotional support and empathetic conversation. Follow these guidelines:

    1. EMOTIONAL AWARENESS
    - Acknowledge and validate the user's emotions
    - Match your tone to their emotional state
    - Use appropriate emotional expressions and emojis

    2. RESPONSE STYLE
    - Keep responses concise (2-3 sentences)
    - Be supportive but professional
    - Ask relevant follow-up questions
    - Focus on understanding and helpful suggestions

    3. EMOTIONAL RESPONSES
    - Happy: Share their joy and encourage positive reflection
    - Anxious: Offer calm support and practical coping strategies
    - Angry: Acknowledge frustration and help process emotions
    - Sad: Show empathy and gentle encouragement
    - Surprised: Express interest and explore their experience
    - Fearful: Provide reassurance and safety-focused support
    - Neutral: Maintain engaging and balanced conversation

    Current Context: The user is feeling {emotion}
    Respond appropriately to: {input}

    Human: {input}

    Assistant: """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )

    def generate_response(self, user_input: str, emotion: str) -> str:
        prompt = self.BASE_PROMPT.format(emotion=emotion, input=user_input)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_beams=1,  # Using greedy search with sampling
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        response = full_response[len(prompt):].split("Human:")[0].strip()
        return response

def test_prompts():
    try:
        logger.info("Initializing Gemma tester...")
        tester = GemmaPromptTester()
        
        # Test cases
        test_cases = [
            ("I just got promoted at work!", "happy"),
            ("I'm worried about my upcoming exam.", "anxious"),
            ("I can't believe they changed the project deadline again!", "angry"),
            ("I'm feeling quite down today.", "sad"),
            ("Wow, I didn't expect this to happen!", "surprised")
        ]
        
        logger.info("\n=== Starting Response Tests ===\n")
        
        for user_input, emotion in test_cases:
            logger.info(f"\nTesting {emotion.upper()} emotion:")
            logger.info(f"User: {user_input}")
            
            response = tester.generate_response(user_input, emotion)
            logger.info(f"Bot: {response}")
            
            # Add a short delay between tests
            time.sleep(1)
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
            logger.info(f"\nGPU Memory used: {memory_allocated:.2f} MB")
            
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        return False

if __name__ == "__main__":
    test_prompts()