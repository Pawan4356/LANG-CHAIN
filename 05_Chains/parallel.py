from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Generate simple and short notes on: \n{text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 3 quiz questions with options on: \n{text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the given notes: {notes} \nAnd quiz: {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser
chain = parallel_chain | merge_chain
result = chain.invoke({'text': 'SVM Models'})

print(result + '\n')

chain.get_graph().print_ascii()

# Output

"""
### SVM (Support Vector Machine) Models  
**Core Idea**:  
- **Classification/Regression Algorithm**: Finds the optimal boundary (hyperplane) that separates data into classes or predicts continuous values.  

#### Key Concepts:  
1. **Hyperplane**:  
   - Decision boundary (line/plane) separating classes.  
   - **Goal**: Maximize the margin (distance) between classes.  

2. **Support Vectors**:  
   - Critical data points closest to the hyperplane.  
   - Define the position/orientation of the hyperplane.  

3. **Margin**:  
   - Gap between support vectors and the hyperplane.  
   - **Hard Margin**: Strict classification (zero errors allowed).  
   - **Soft Margin**: Allows controlled misclassifications (via `C` parameter).  

#### Types:  
1. **Linear SVM**:  
   - Separates linearly separable data with a straight line/plane.  
   - *Example*: Classifying fruits by size/weight.  

2. **Non-linear SVM**:  
   - Uses the **Kernel Trick** to handle complex data.  
   - Projects data into higher dimensions for easier separation.  
   - **Common Kernels**:  
     - `Polynomial`: Creates curved boundaries.  
     - `RBF` (Radial Basis Function): Forms circular/radial boundaries (most versatile).  

#### Optimization & Formula:  
- **Objective**: Minimize \( \frac{1}{2} \|\mathbf{w}\|^2 + C \sum \xi_i \)  
  where \( \mathbf{w} \) defines the hyperplane, \( \xi_i \) are slack variables (errors), and `C` balances margin width vs. misclassification tolerance.  
- **Kernel Trick**: Efficiently computes dot products \(\langle \phi(x_i), \phi(x_j) \rangle\) without explicit high-dimensional transformation.  
  - *Example*: RBF kernel: \( K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2) \).  

#### Pros & Cons:  
- ✅ **Effective** in high-dimensional spaces.  
- ✅ Robust with clear margin separation.  
- ❌ Computationally heavy for large datasets.  
- ❌ Requires tuning `C` and kernel parameters.  

#### Use Cases:  
- Image/text classification, bioinformatics (gene classification), handwriting recognition.  

---

### Review Questions  
**Question 1**: What is the primary goal of SVM optimization?  
- **Answer**: b) Maximize the margin between classes  
- *Explanation*: SVMs prioritize maximizing the margin distance to the nearest data points for robust separation.  

**Question 2**: What is the kernel trick's role in SVM?  
- **Answer**: b) Computes similarity in high-dimensional space without explicit transformation  
- *Explanation*: Kernels (e.g., RBF) implicitly map data to higher dimensions, enabling linear separation of complex patterns.  

**Question 3**: What does hyperparameter `C` control in soft-margin SVM?  
- **Answer**: b) Trade-off between misclassification tolerance and margin width  
- *Explanation*: High `C` prioritizes accuracy (smaller margin, fewer errors); low `C` allows more errors for a wider margin.  

**Key Concepts Reinforced**:  
- Margin maximization as the SVM optimization objective.  
- Kernel trick for efficient non-linear separation.  
- Regularization via `C` to balance error tolerance and margin width.
"""