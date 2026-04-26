from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Provide a detailed summary of {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Provide a 5 line summary for: {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic": "Gen-AI"})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result1.content})
result2 = model.invoke(prompt2)

print(f"Detailed info:\n\n {result1.content} \n\n Summary: \n\n {result2.content}")

# Output

"""
Detailed info:

 ## **Generative AI (Gen-AI): A Detailed Summary**

### **1. Core Definition**
**Generative AI** refers to a subset of artificial intelligence systems capable of creating new, original content—such as text, images, audio, video, code, or 3D models—by learning patterns from existing data. Unlike traditional AI, which focuses on classification or prediction, Gen-AI produces novel outputs that often resemble human-created content.

---

### **2. Key Technologies & Architectures**
Gen-AI is powered by advanced machine learning models, primarily:
- **Large Language Models (LLMs)**: Transformer-based models (e.g., GPT-4, Claude, Gemini) trained on vast text corpora to generate coherent language.
- **Diffusion Models**: Used for image generation (e.g., DALL-E, Stable Diffusion), these models gradually add and remove noise to create high-quality visuals.
- **Generative Adversarial Networks (GANs)**: Two neural networks (generator and discriminator) compete to produce realistic synthetic data.
- **Variational Autoencoders (VAEs)**: Encode data into a latent space and decode it to generate new samples.
- **Multimodal Models**: Combine text, image, audio, and other data types (e.g., OpenAI’s Sora for video, Midjourney for images).

---

### **3. How It Works: The Training Process**
1. **Data Collection**: Models are trained on massive datasets (e.g., internet text, image libraries).
2. **Pattern Learning**: Neural networks identify underlying structures, relationships, and probabilities within the data.
3. **Generation**: Given a prompt or seed input, the model predicts plausible next elements (words, pixels, etc.) based on learned patterns.

---

### **4. Major Applications & Use Cases**
- **Content Creation**: Writing articles, marketing copy, social media posts, and creative stories.
- **Visual Arts & Design**: Generating logos, illustrations, concept art, and product designs.
- **Audio & Music**: Composing music, voice synthesis, and sound effect generation.
- **Code Generation**: Automating software development (e.g., GitHub Copilot).
- **Scientific Research**: Accelerating drug discovery, material science, and hypothesis generation.
- **Gaming & Virtual Worlds**: Creating immersive environments, characters, and narratives.
- **Personalized Marketing**: Tailoring ads, emails, and recommendations dynamically.

---

### **5. Societal & Economic Impact**
- **Productivity Boost**: Automates repetitive creative tasks, augmenting human capabilities.
- **Democratization of Creativity**: Lowers barriers for non-experts to produce high-quality content.
- **Industry Disruption**: Impacts sectors like media, education, entertainment, and software development.
- **Job Market Shifts**: May displace some roles while creating new opportunities in AI oversight, prompt engineering, and ethics.

---

### **6. Critical Challenges & Ethical Concerns**
- **Bias & Fairness**: Models can perpetuate societal biases present in training data.
- **Misinformation**: Potential for generating convincing fake news, deepfakes, or plagiarized content.
- **Intellectual Property**: Unclear legal frameworks around AI-generated content ownership.
- **Environmental Cost**: Training large models consumes significant energy and resources.
- **Accountability**: Difficulty in attributing responsibility for harmful or erroneous outputs.
- **Overreliance**: Risk of eroding human skills and critical thinking.

---

### **7. Future Directions**
- **Improved Multimodality**: Seamless integration of text, audio, video, and sensory data.
- **Real-Time Generation**: Faster, more efficient models for interactive applications.
- **Personalized AI**: Tailored models trained on individual preferences and data.
- **Regulation & Governance**: Developing global standards for ethical deployment.
- **AI-Human Collaboration**: Enhanced tools for co-creation and augmentation.

---

### **8. Leading Players & Tools**
- **OpenAI**: GPT series, DALL-E, Sora.
- **Google**: Gemini, Imagen.
- **Meta**: Llama models.
- **Anthropic**: Claude.
- **Stability AI**: Stable Diffusion.
- **Midjourney**: Image generation.
- **Numerous startups** and open-source communities.

---

### **Conclusion**
Generative AI represents a paradigm shift in how humans interact with technology, blurring the lines between human and machine creativity. While it offers transformative potential across industries, its responsible development requires careful attention to ethical, legal, and societal implications. As the technology evolves, the focus will likely shift from mere content generation to fostering trustworthy, equitable, and collaborative AI systems. 

 Summary: 

 Generative AI creates original content like text and images by learning from data. It uses models such as LLMs and diffusion models for tasks ranging from writing to design. While it boosts productivity and democratizes creation, it also raises ethical concerns like bias and misinformation. Its rapid evolution is impacting numerous industries and the job market. Responsible development is crucial to harness its potential ethically.
"""