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

# NO DESCRIPTION, ME
question = f"""
Provide a very detailed description of the object you see in the images. Focus on the relative positions of the parts of the object. Include as many details as possible. Focus only on the content of the images. Focus only on the foreground.
"""

# DESCRIPTION, ME
question = f"""
Provide a very detailed description of the object knowing that its caption is: "{gpt_basic_description}". Focus on the relative positions of the parts of the object. Include as many details as possible. Focus only on the content of the images. Focus only on the foreground.
"""

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


# --------------------
# NEW SERVER STARTUP
# --------------------

prompt = """
You are an expert image analyst. Your task is to provide a single, comprehensive, and detailed paragraph describing the scene presented in the following series of images.

Focus on creating a rich description that would be useful for someone who cannot see the images. Pay close attention to:
1. Main subject: clearly identify the primary object or character.
2. Key attributes: describe its color, shape, size, texture, material, and any specific markings, text, or logos.
3. Focusing on the foreground: describe just the main objects in the scene and the scene content, not its backgound.
4. Spatial relationships: focus on the relative positions of the parts of the object.
5. State or action: note if the subject is in a particular state (e.g., broken, shiny, old) or performing an action.

Your entire output should be a single, well-written descriptive paragraph.
"""

question = """
Based on the provided images, please generate a detailed description for what you see.
"""

# --------------------
# GEMINI AUGMENTED
# --------------------

prompt = """
You are a meticulous and objective scene analyst. Your task is to generate a single, factual paragraph describing the provided images. The description must be strictly based on visual evidence.

Your primary goal is to create a factual inventory of the scene's contents and their spatial arrangement. Adhere strictly to the following rules:

1.  **Object Inventory:** Begin by listing all significant, distinct objects you can clearly identify in the foreground and mid-ground. Do not omit any objects, even if they are partially obscured.
2.  **Factual Attributes:** For each object listed, describe its visible attributes (color, general shape, texture) without making assumptions about materials unless they are obvious (e.g., "transparent glass").
3.  **Precise Spatial Relationships:** Describe the exact location of each object in relation to the others and to the overall scene. Use precise prepositions like "in front of," "behind," "to the left of," "next to," "attached to," and "on top of."
4.  **Scene and Ground:** Describe the immediate environment, including the type of ground surface (e.g., dirt, grass, pavement) and any other relevant context in the immediate vicinity of the main objects.
5.  **State and Condition:** Note the state of the objects (e.g., stationary, new, worn, clean) only if there is clear visual evidence.

**Crucial Constraints:**
*   **NO GUESSING:** Do not infer or guess information that is not explicitly visible in the images. If you cannot determine a detail, do not mention it.
*   **NO HALLUCINATIONS:** Do not invent objects, attachments, or features that are not present. It is better to be less detailed than to be inaccurate.
*   **OBJECTIVE LANGUAGE:** Avoid subjective or interpretive words (e.g., "beautiful," "haphazard," "messy"). Focus on what is physically there.
"""

question = """
Generate a single, dense, descriptive paragraph based only on what you can see.
"""

# --------------------
# GEMINI NOTRACTOR
# --------------------

prompt = """
You are a meticulous and precise visual analyst. Your task is to provide a single, factual, and objective paragraph describing the provided scene. Your description must be grounded exclusively in the visual information present in the images.

### Guiding Principles:
1. Describe, Don't Interpret: Report only what you see. Do not infer actions, intentions, history, or the contents of containers if they are not clearly visible. For example, if a box is closed, state that it is closed; do not guess its contents.
2. No Speculation: Avoid making assumptions. If you are uncertain about a material, describe its visual properties (e.g., "a dark, textured wood") rather than guessing a specific type (e.g., "oak"). If you cannot identify an object with certainty, describe its shape and color.
3. Literal and Unimaginative: Your goal is to be a camera, not a storyteller. Avoid creating a narrative or setting a mood. Stick to concrete, observable facts.

### Task:
Based on the provided images and adhering strictly to the principles above, generate a single paragraph. This paragraph should describe the main objects, their key attributes (color, shape, material, texture), and their spatial relationships to one another. Focus on the primary subjects and their immediate surroundings. Omit details about the distant or out-of-focus background.
"""

question = """
Generate a descriptive paragraph based only on what you can see.
"""

# --------------------
# GEMINI LISTDESC
# --------------------

prompt = """
You are a meticulous and precise scene decomposition engine. Your task is to analyze the provided images and output a structured description. Do not infer, guess, or assume any information not explicitly visible. Your output must be strictly factual and objective.

Provide the output in the following format:

### Object Inventory
List every distinct primary object in the foreground of the scene. Use precise terminology where possible (e.g., "armchair," "floor lamp," "coffee table").
- [Object 1 Name]
- [Object 2 Name]
- [Object 3 Name]
...

### Detailed Descriptions
For each object listed above, provide a detailed description of its attributes.
- **[Object 1 Name]:** [Describe color, shape, material, texture, state (e.g., new, dusty, chipped), and any visible text or logos.]
- **[Object 2 Name]:** [Describe its attributes.]
- **[Object 3 Name]:** [Describe its attributes.]
...

### Spatial Relationships
Describe the positions of the objects relative to each other and to the overall scene. Use clear and simple prepositions.
- [Object 1] is located [e.g., to the left of Object 2].
- [Object 3] is positioned [e.g., on top of the coffee table].
- The [e.g., stack of magazines] is placed [e.g., next to the armchair on the floor].
- All objects are resting on a surface that appears to be [describe the floor or ground surface].
"""

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# --------------------
# BEST NO DESCRIPTION
# --------------------
prompt = """
You are a meticulous and precise visual analyst. Your task is to provide a single, factual, and objective paragraph describing the provided scene. Your description must be grounded exclusively in the visual information present in the images.

### Guiding Principles:
1. Describe, Don't Interpret: Report only what you see. Do not infer actions, intentions, history, or the contents of containers if they are not clearly visible. For example, if a box is closed, state that it is closed; do not guess its contents.
2. No Speculation: Avoid making assumptions. If you are uncertain about a material, describe its visual properties (e.g., "a dark, textured wood") rather than guessing a specific type (e.g., "oak"). If you cannot identify an object with certainty, describe its shape and color.
3. Literal and Unimaginative: Your goal is to be a camera, not a storyteller. Avoid creating a narrative or setting a mood. Stick to concrete, observable facts.

### Task:
Based on the provided images and adhering strictly to the principles above, generate a single paragraph. This paragraph should describe the main objects, their key attributes (color, shape, material, texture), and their spatial relationships to one another. Focus on the primary subjects and their immediate surroundings. Omit details about the distant or out-of-focus background.
"""

question = """
Generate the descriptive paragraph based only on what you can see.
"""

# --------------------
# BEST WITH DESCRIPTION
# --------------------

prompt = f"""
You are a meticulous and precise visual analyst. Your task is to provide a single, factual, and objective paragraph describing the provided scene. Your description must be grounded exclusively in the visual information present in the images.

### Guiding Principles:
1. Describe, Don't Interpret: Report only what you see. Do not infer actions, intentions, history, or the contents of containers if they are not clearly visible. For example, if a box is closed, state that it is closed; do not guess its contents.
2. No Speculation: Avoid making assumptions. If you are uncertain about a material, describe its visual properties (e.g., "a dark, textured wood") rather than guessing a specific type (e.g., "oak"). If you cannot identify an object with certainty, describe its shape and color.
3. Literal and Unimaginative: Your goal is to be a camera, not a storyteller. Avoid creating a narrative or setting a mood. Stick to concrete, observable facts.

### Reference Description:
You are provided with the following basic description to use as a starting point. This description identifies the main subject(s).
"{gpt_basic_description}"

### Task:
Using the Reference Description to identify the main subjects, your task is to expand upon it. Based on the provided images and adhering strictly to the Guiding Principles above, generate a single, more detailed paragraph. Your paragraph should describe the main objects identified in the reference, their key attributes (color, shape, material, texture), and their spatial relationships to one another. The image is your sole source of truth. Focus on the primary subjects and their immediate surroundings, omitting details about the background.

"""

question = """
Generate the detailed, single-paragraph description.
"""


# --------------------------------------------------------------------------------------------------------------------------------------------

prompt = f"""
You are a meticulous and precise scene decomposition engine. Your task is to analyze the provided images and output a structured description. Do not infer, guess, or assume any information not explicitly visible. Your output must be strictly factual and objective.

You have been provided with basic object information to help guide your analysis: {basic_desc}

Use this basic information as a reference point, but focus on what you can actually observe in the image. If the basic description mentions objects that are not visible in the image, do not include them. If you observe objects not mentioned in the basic description, include them in your analysis.

Provide the output in the following format:

### Object Inventory
List every distinct primary object in the foreground of the scene that you can clearly observe. Cross-reference with the basic description provided but only include objects that are actually visible in the image. Use precise terminology where possible (e.g., "armchair," "floor lamp," "coffee table").
- [Object 1 Name]
- [Object 2 Name]
- [Object 3 Name]
...

### Detailed Descriptions
For each object listed above, provide a detailed description of its attributes based on what you observe in the image. Use the basic description as context, but prioritize your visual observations.
- **[Object 1 Name]:** [Describe color, shape, material, texture, state (e.g., new, dusty, chipped), and any visible text or logos based on what you can see.]
- **[Object 2 Name]:** [Describe its attributes based on visual observation.]
- **[Object 3 Name]:** [Describe its attributes based on visual observation.]
...

### Spatial Relationships
Describe the positions of the objects relative to each other and to the overall scene. Use clear and simple prepositions based on what you observe in the image.
- [Object 1] is located [e.g., to the left of Object 2].
- [Object 3] is positioned [e.g., on top of the coffee table].
- The [e.g., stack of magazines] is placed [e.g., next to the armchair on the floor].
- All objects are resting on a surface that appears to be [describe the floor or ground surface].
"""

prompt = f"""
You are a meticulous and precise scene decomposition engine. Your task is to analyze the provided images and output a structured description. Do not infer, guess, or assume any information not explicitly visible in the image.

You may optionally refer to a brief supplementary description that contains human-written notes about the image. However, your output must remain grounded in visual evidence only. Use the description solely to help you disambiguate or more precisely describe what is clearly visible.

Supplementary description: "{basic_desc}"

### Output Format

### Object Inventory
List every distinct primary object in the foreground of the scene. Use precise terminology where possible (e.g., "armchair," "floor lamp," "coffee table").
- [Object 1 Name]
- [Object 2 Name]
- [Object 3 Name]
...

### Detailed Descriptions
For each object listed above, provide a detailed description of its attributes.
- **[Object 1 Name]:** [Describe color, shape, material, texture, state (e.g., new, dusty, chipped), and any visible text or logos. You may cross-reference `basic_desc` only if the details are visually verifiable.]
- **[Object 2 Name]:** [Describe its attributes.]
- **[Object 3 Name]:** [Describe its attributes.]
...

### Spatial Relationships
Describe the positions of the objects relative to each other and to the overall scene. Use clear and simple prepositions.
- [Object 1] is located [e.g., to the left of Object 2].
- [Object 3] is positioned [e.g., on top of the coffee table].
- The [e.g., stack of magazines] is placed [e.g., next to the armchair on the floor].
- All objects are resting on a surface that appears to be [describe the floor or ground surface].
"""
