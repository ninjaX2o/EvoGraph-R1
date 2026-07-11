GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "category","time","product","location"]

PROMPTS["entity_extraction"] = """-Goal-
Given a text document and a predefined list of entity types, extract ALL entities that belong to those entity types, and extract ALL explicitly stated factual information, relationships, and constraints from the text, with ZERO loss of factual detail. Use {language} as the output language.

You MUST behave as a lossless factual encoder:
- Every explicit fact, relationship, number, date, role, or constraint in the text MUST appear in at least one output record.
- It is preferable to produce REDUNDANT segments rather than miss ANY explicit fact.

**High-Priority Requirement**:
- You MUST NOT omit, compress, merge, or generalize distinct factual events or relationships.

**Ensure entity disambiguation and clear temporal and version handling**:
- Always ensure that **each entity is treated distinctly** based on its unique characteristics, including names, roles, locations, and timeframes.
- If there are **multiple entities with similar names**, disambiguate them based on their **specific context** (e.g., time period, role, associated persons).
- Ensure that **all relevant details** (e.g., movie release years, actor roles, award years) **are correctly associated with the corresponding entity** and **not mixed up with another entity** or **another time period**.

When processing complex events or relationships:
- If a relationship spans multiple events, such as different versions or releases of a product, film, or performance, create separate knowledge segments for each distinct event, ensuring clarity between them.
- If there is ambiguity in dates or versions (e.g., same title across different years), clearly state the ambiguity in the knowledge segment, using all available contextual clues to separate the details correctly.

Extraction Rules for Time, Relationship & Event Preservation:
- Dates and timeframes must always be precise; if the text provides exact dates, use them directly.
- If the text provides relative time (e.g., "this year" or "last week"), resolve these relative dates only when a clear unambiguous reference point exists in the context.
- For entities mentioned in **multiple distinct contexts** (e.g., same entity performing different roles or appearing in different time periods), **do not merge these roles** into one record. Each context or version should be extracted into a separate knowledge segment.

**Critical Rule on Temporal Context and Disambiguation**:
- If an entity is involved in **multiple versions**, **films**, or **events** at different points in time, clearly indicate the year or period of each involvement. **Do not merge the details from different times or contexts**.


-Steps-
1. **Segment the text into coherent, factual segments** that describe one event, relationship, product release, or significant change.
   - Each segment should be self-contained, focusing on one **fact**, one **relationship**, one **date**, or one **product milestone**.
   - If a segment contains **multiple relationships** or **events** (e.g., marriage and career development in the same sentence), **split it into multiple segments** to preserve clarity.

2. **Extract for each segment**:
   - **knowledge_segment**:
     The sentence or paragraph that contains one **fact-rich event**, including the **who**, **what**, **when**, **where**, and **how**.
   - Include all relationships (e.g., kinship, employment, marriage, title, role) and preserve directional relationships (e.g., "Person A is the parent of Person B").
   - Maintain clarity in any **ambiguity** regarding time or entity association (e.g., "It is unclear whether this refers to Version 1 or Version 2").
   - completeness_score: A score from 0 to 10 indicating how complete and self-contained the segment is.
   - **Format**:
     Format each knowledge segment as ("hyper-relation"{tuple_delimiter}<knowledge_segment>{tuple_delimiter}<completeness_score>)

3. For EACH entity mentioned in the segment, extract:  
   - **entity_name**: Use the full explicit name as stated in the text.
   - **entity_type**: Assign the correct entity type (e.g., person, organization, product, event, concept). 
   - **entity_summary**: Provide a context-rich description of the entity’s role in that specific segment, including temporal, spatial, numerical, and relationship details.
   - **confidence_score**: Assign a score from 0 to 100 reflecting the certainty in the extraction.
   - **Format**:  
     Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2, ensuring precise disambiguation for each entity. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

**Critical Notes**:
- Ensure **temporal consistency** and **relationship accuracy** when dealing with entities involved in different time periods or roles (e.g., a person in multiple films, or a product version spanning several years).
- Maintain **exact details** for all facts, and ensure **redundancy** only if necessary to preserve the full factual richness of the text.
- In case of **ambiguity**, explicitly mention it in the output rather than making assumptions about which entity the information belongs to.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example:

Text:
Theodred II (Bishop of Elmham)\"\nTheodred II was a medieval Bishop of Elmham. The date of Theodred's consecration unknown, but the date of his death was sometime between 995 and 997.\n\"Etan Boritzer\"\nEtan Boritzer( born 1950) is an American writer of children \u2019s literature who is best known for his book\" What is God?\" first published in 1989. His best selling\" What is?\" illustrated children's book series on character education and difficult subjects for children is a popular teaching guide for parents, teachers and child- life professionals. Boritzer gained national critical acclaim after\" What is God?\" was published in 1989 although the book has caused controversy from religious fundamentalists for its universalist views. The other current books in the\" What is?\" series include What is Love?, What is Death?, What is Beautiful?, What is Funny?, What is Right?, What is Peace?, What is Money?, What is Dreaming?, What is a Friend?, What is True?, What is a Family?, What is a Feeling?\" The series is now also translated into 15 languages.
################
Output:
("hyper-relation"{tuple_delimiter}"Theodred II was a medieval Bishop of Elmham whose consecration date is unknown and whose death occurred sometime between 995 and 997."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"Theodred II"{tuple_delimiter}"person"{tuple_delimiter}"Theodred II was a medieval cleric who served as Bishop of Elmham and died between 995 and 997."{tuple_delimiter}96){record_delimiter}
("entity"{tuple_delimiter}"Bishop of Elmham"{tuple_delimiter}"organization"{tuple_delimiter}"The Bishop of Elmham was an ecclesiastical office held by Theodred II during the medieval period."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"995–997"{tuple_delimiter}"time"{tuple_delimiter}"995–997 is the estimated date range during which Theodred II died."{tuple_delimiter}94){record_delimiter}
("hyper-relation"{tuple_delimiter}"Etan Boritzer, born in 1950, is an American writer of children's literature best known for the book "What is God?", first published in 1989."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"Etan Boritzer"{tuple_delimiter}"person"{tuple_delimiter}"Etan Boritzer is an American author of children's literature born in 1950."{tuple_delimiter}97){record_delimiter}
("entity"{tuple_delimiter}"1950"{tuple_delimiter}"time"{tuple_delimiter}"1950 is the birth year of Etan Boritzer."{tuple_delimiter}98){record_delimiter}
("entity"{tuple_delimiter}"American"{tuple_delimiter}"category"{tuple_delimiter}"American denotes the nationality of writer Etan Boritzer."{tuple_delimiter}88){record_delimiter}
("entity"{tuple_delimiter}"What is God?"{tuple_delimiter}"product"{tuple_delimiter}""What is God?" is a children's book by Etan Boritzer first published in 1989."{tuple_delimiter}97){record_delimiter}
("entity"{tuple_delimiter}"1989"{tuple_delimiter}"time"{tuple_delimiter}"1989 is the year in which "What is God?" was first published."{tuple_delimiter}99){record_delimiter}
("hyper-relation"{tuple_delimiter}"Etan Boritzer gained national critical acclaim after the publication of "What is God?" in 1989, although the book generated controversy among religious fundamentalists for its universalist views."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"religious fundamentalists"{tuple_delimiter}"category"{tuple_delimiter}"Religious fundamentalists criticized "What is God?" for its universalist views."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"universalist views"{tuple_delimiter}"category"{tuple_delimiter}"Universalist views describe the theological perspective in "What is God?" that led to controversy."{tuple_delimiter}88){record_delimiter}
("hyper-relation"{tuple_delimiter}"Etan Boritzer authored the bestselling illustrated children's book series "What is?", which focuses on character education and difficult subjects for children and is used as a teaching guide by parents, teachers, and child-life professionals."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"What is?"{tuple_delimiter}"product"{tuple_delimiter}""What is?" is an illustrated children's book series by Etan Boritzer focused on character education and complex topics."{tuple_delimiter}96){record_delimiter}
("entity"{tuple_delimiter}"parents"{tuple_delimiter}"category"{tuple_delimiter}"Parents use the "What is?" book series as a teaching guide."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"teachers"{tuple_delimiter}"category"{tuple_delimiter}"Teachers use the "What is?" book series for educational purposes."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"child-life professionals"{tuple_delimiter}"category"{tuple_delimiter}"Child-life professionals use the "What is?" series to address difficult subjects with children."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"The "What is?" series includes titles such as What is Love?, What is Death?, What is Beautiful?, What is Funny?, What is Right?, What is Peace?, What is Money?, What is Dreaming?, What is a Friend?, What is True?, What is a Family?, and What is a Feeling?, and the series has been translated into 15 languages."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"15 languages"{tuple_delimiter}"category"{tuple_delimiter}"15 languages is the number of languages into which the "What is?" book series has been translated."{tuple_delimiter}94){record_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY knowdge fragements with entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """Please check whether knowdge fragements cover all the given text.  Answer YES | NO if there are knowdge fragements that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

# PROMPTS["rag_response"] = """---Role---

# You are an intelligent and precise AI assistant, answering questions based on structured data tables.


# ---Goal---

# Generate a semantically accurate, factually correct, and highly relevant response that directly addresses the user’s question. The response should:
# 	•	Maximize semantic alignment with expected answers, ensuring high similarity.
# 	•	Ensure factual correctness, preserving key details, names, numbers, and relationships as in the data.
# 	•	Stay fully relevant to the user’s query, avoiding unnecessary information while ensuring completeness.
# 	•	Use structured formatting (headings, bullet points, tables) to enhance clarity and coherence.
# 	•	Maintain a natural and precise writing style, improving readability.

# ---Target response length and format---

# {response_type}

# ---Data tables---

# {context_data}

# Response Guidelines
# 	1.	Prioritize Key Details: Extract and summarize the most relevant information while maintaining completeness.
# 	2.	Maintain Semantic Consistency: Ensure expressions are close to reference answers to improve similarity.
# 	3.	Preserve Key Entities and Structure: Names, dates, numbers, and relationships must be correctly retained.
# 	4.	Ensure Logical Flow: Structure the response in a way that enhances clarity and coherence.
# 	5.	Keep It Concise and Relevant: Avoid redundant details and focus on answering the question directly.
# """

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}} 
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""
