# WORKS PRETTY BAD, I THOUGHT BETTER

question = f"""
You are an AI assistant tasked with generating detailed multiple-choice questions and their corresponding answers based on the images provided.

### Task:
Generate one detailed multiple-choice question about the object(s) or scene depicted in the provided images.

### Constraints:
1.  **Relative References Only:** Since the scene can be viewed from multiple perspectives and may not have a fixed orientation, all questions and options must use *only* relative points of reference (e.g., "to the left of [object A]", "above [object B]", "between [object X] and [object Y]").
2.  **Detailed Questions:** The questions should require careful observation of the image, and should be about many different aspects of the image.
3.  **Distinct Options:** Provide three distinct options for each question.
4.  **Correct Answer:** One option must be clearly correct based on the image.

### Output Format:
For each question-answer pair, you MUST follow this format precisely:

Q: [Your Question Text Here]
1. [Option 1 Text]
2. [Option 2 Text]
3. [Option 3 Text]
A: [Correct Option Number]. [Full Text of Correct Option]

### Example of Output Format:
(Imagine an image shows a red apple to the left of a green pear)

Q: What fruit is located to the immediate right of the red apple?
1. A blue banana
2. A green pear
3. An orange orange
A: 2. A green pear

---
Now, based on the provided images and the object description: "{cap[0]['conversations'][1]['value']}", generate one question and answer following all the above instructions and the specified format.
"""


# REMOVED THE EXAMPLE WHICH IS NOT RELATED SINCE IT DOESN'T HAVE THE IMAGE
question = f"""
You are an AI assistant tasked with generating detailed multiple-choice questions and their corresponding answers based on the images provided.

### Task:
Generate one detailed multiple-choice question about the object(s) or scene depicted in the provided images.

### Constraints:
1.  **Detailed Questions:** The questions should require careful observation of the image, and should be about many different aspects of the image.
2.  **Distinct Options:** Provide three distinct options for each question.
3.  **One Correct Answer:** Only one option must be clearly correct based on the image.

### Output Format:
For each question-answer pair, you MUST follow this format precisely:

Q: [Your Question Text Here]
1. [Option 1 Text]
2. [Option 2 Text]
3. [Option 3 Text]
A: [Correct Option Number]. [Full Text of Correct Option]

Now, based on the provided images and the object description: "{cap[0]['conversations'][1]['value']}", generate a question and the relative answer following all the above instructions and the specified format.
"""

# ===================================================================================================================================================================
# ===================================================================================================================================================================
#                                                                     DETAILED DESCRIPTIONS
# ===================================================================================================================================================================
# ===================================================================================================================================================================

# NO DESCRIPTION, GEMINI 2.5 PRO
question = """
You are an expert image analyst. Your task is to provide a single, comprehensive, and detailed paragraph describing the scene presented in the following series of images.

Focus on creating a rich description that would be useful for someone who cannot see the images. Pay close attention to:

1.  **Main Subject:** Clearly identify the primary object or character.
2.  **Key Attributes:** Describe its color, shape, size, texture, material, and any specific markings, text, or logos.
3.  **Setting & Background:** Describe the environment where the subject is located. What is around it?
4.  **Spatial Relationships:** Explain how different elements in the scene are positioned relative to each other (e.g., "the object is on top of the table," "to the left of the building," "partially obscured by a tree").
5.  **State or Action:** Note if the subject is in a particular state (e.g., broken, shiny, old) or performing an action.

**IMPORTANT:** Do NOT create a question, a list, or a QnA. Your entire output should be a single, well-written descriptive paragraph.

Now, based on the images provided, generate your detailed description.
"""

# DESCRIPTION, GEMINI 2.5 PRO
question = """
You are an AI assistant specializing in description enhancement. You will be given a set of images and a brief, human-provided description.

Your task is to use the brief description as a starting point and EXPAND it into a much more detailed and comprehensive paragraph by meticulously analyzing the visual information in the images.

Your goal is to add new, valuable details that are not present in the original brief description. Focus on:

*   **Elaborating on the main subject:** Add specifics about its color, texture, material, and condition.
*   **Detailing the background:** Describe the environment, time of day, and other surrounding objects.
*   **Specifying spatial relationships:** Clearly state where things are in relation to each other.
*   **Adding interesting observations:** Note any unique features, text, logos, or unusual aspects visible in the images.

**Example Input Format:**
BRIEF DESCRIPTION: A car with a hat on it.
[Your enhanced, detailed paragraph would go here, describing the type of car, its color, the type of hat, the street, etc.]

**IMPORTANT:** Do NOT just repeat the brief description. Your primary job is to ADD information from the images. Do not write a question or a list. Your output must be a single, enhanced paragraph.

---
**YOUR TASK**

**BRIEF DESCRIPTION:** [Here you will insert the brief human annotation]

Now, using this brief description and the images provided, generate your enhanced, detailed description.
"""

# NO DESCRIPTION, GPT 4o
question = """
Describe the image in detail. Focus on what can be clearly seen.

Include:
- Object appearance (color, shape, texture, material)
- Spatial relationships between visible parts (e.g., “on the left”, “next to”)
- Any visible components, patterns, or symbols

Do not guess the function or identity unless it's visually obvious.
Use neutral, objective language.

Output:
Description: <your detailed visual description here>
"""

# DESCRIPTION, GPT 4o
question = """
You are given an image and a short human annotation.

Annotation: "<brief annotation here>"

Use both the image and the annotation to produce a more detailed visual description. Expand and clarify based on what is visible in the image. Do not copy or repeat the annotation directly.

Focus on:
- Visual features (shape, color, texture, material)
- Relative positions and structure
- Any visible text, symbols, or labels

Output:
Description: <your detailed visual description here>
"""