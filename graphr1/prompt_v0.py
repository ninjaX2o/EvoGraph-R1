GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "category","time","product","location"]

PROMPTS["entity_extraction"] = """-Purpose-
Extract structured high-order relational knowledge from raw text documents by identifying semantically coherent segments and their associated entities.

-Goal-
Given a paragraph or document and a predefined list of entity types, identify knowledge segments and extract all textual entities associated with each segment.
Use {language} as output language.

-Steps-
1. Divide the text into the minimum number of self-contained, fact-rich segments. Each segment must be independently interpretable without reading surrounding text.  For each knowledge segment, extract the following information:
-- knowledge_segment: the self-contained factual text.
-- completeness_score: A score from 0 to 10 indicating how complete and self-contained the segment is.
Format each knowledge segment as ("hyper-relation"{tuple_delimiter}<knowledge_segment>{tuple_delimiter}<completeness_score>)

2. Identify all entities in each knowledge segment. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name. Use the full explicit name as stated in the text.
- entity_type: Type of the entity.
- entity_description: Provide a context-rich description of the entity's role in that specific segment, including temporal, spatial, numerical, and relationship details.
- key_score: A score from 0 to 100 indicating the importance of the entity in the text.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)

3. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

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
    """Example 1:

Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order. Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.” The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce. It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("hyper-relation"{tuple_delimiter}Theodred II was a medieval Bishop of Elmham; his consecration date is unknown, and his death occurred sometime between 995 and 997.{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}Theodred II{tuple_delimiter}PERSON{tuple_delimiter}Theodred II was a medieval religious figure identified as Bishop of Elmham, with an unknown consecration date and a death dated to sometime between 995 and 997.{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}Bishop of Elmham{tuple_delimiter}RELIGIOUS_TITLE{tuple_delimiter}Bishop of Elmham was the medieval ecclesiastical office held by Theodred II.{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}Elmham{tuple_delimiter}PLACE{tuple_delimiter}Elmham is the location associated with the bishopric held by Theodred II in the medieval period.{tuple_delimiter}70){record_delimiter}
("entity"{tuple_delimiter}995 and 997{tuple_delimiter}DATE_RANGE{tuple_delimiter}The years 995 and 997 define the estimated range within which Theodred II died.{tuple_delimiter}75){record_delimiter}
("hyper-relation"{tuple_delimiter}Etan Boritzer, born in 1950, is an American writer of children’s literature best known for the book What is God?, first published in 1989.{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}Etan Boritzer{tuple_delimiter}PERSON{tuple_delimiter}Etan Boritzer is an American children’s literature writer, born in 1950, whose best-known work is What is God?, first published in 1989.{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}1950{tuple_delimiter}DATE{tuple_delimiter}1950 is the birth year of Etan Boritzer.{tuple_delimiter}65){record_delimiter}
("entity"{tuple_delimiter}American{tuple_delimiter}NATIONALITY{tuple_delimiter}American describes Etan Boritzer’s nationality in the context of his career as a children’s literature writer.{tuple_delimiter}55){record_delimiter}
("entity"{tuple_delimiter}Children’s Literature{tuple_delimiter}GENRE{tuple_delimiter}Children’s literature is the literary field in which Etan Boritzer writes and gained recognition.{tuple_delimiter}70){record_delimiter}
("entity"{tuple_delimiter}What is God?{tuple_delimiter}BOOK{tuple_delimiter}What is God? is Etan Boritzer’s best-known children’s book, first published in 1989 and later associated with both national acclaim and controversy.{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}1989{tuple_delimiter}DATE{tuple_delimiter}1989 is the year when What is God? was first published.{tuple_delimiter}70){record_delimiter}
("hyper-relation"{tuple_delimiter}Etan Boritzer’s best-selling What is? illustrated children’s book series addresses character education and difficult subjects for children, and it is used as a teaching guide by parents, teachers, and child-life professionals.{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}What is?{tuple_delimiter}BOOK_SERIES{tuple_delimiter}What is? is Etan Boritzer’s best-selling illustrated children’s book series focused on character education and difficult subjects for children.{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}Character Education{tuple_delimiter}SUBJECT{tuple_delimiter}Character education is one of the main educational themes addressed by Etan Boritzer’s What is? illustrated children’s book series.{tuple_delimiter}80){record_delimiter}
("entity"{tuple_delimiter}Difficult Subjects For Children{tuple_delimiter}SUBJECT{tuple_delimiter}Difficult subjects for children are a central thematic focus of the What is? series, making it useful for educational and caregiving contexts.{tuple_delimiter}80){record_delimiter}
("entity"{tuple_delimiter}Parents{tuple_delimiter}GROUP{tuple_delimiter}Parents are one of the audiences that use the What is? series as a teaching guide for children.{tuple_delimiter}65){record_delimiter}
("entity"{tuple_delimiter}Teachers{tuple_delimiter}GROUP{tuple_delimiter}Teachers are one of the professional groups using the What is? series as a teaching guide in educational settings.{tuple_delimiter}65){record_delimiter}
("entity"{tuple_delimiter}Child-life Professionals{tuple_delimiter}GROUP{tuple_delimiter}Child-life professionals use the What is? series as a teaching guide for explaining difficult subjects to children.{tuple_delimiter}65){record_delimiter}
("hyper-relation"{tuple_delimiter}Etan Boritzer gained national critical acclaim after What is God? was published in 1989, although the book caused controversy among religious fundamentalists because of its universalist views.{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}Etan Boritzer{tuple_delimiter}PERSON{tuple_delimiter}Etan Boritzer gained national critical acclaim after the 1989 publication of What is God?, while also becoming associated with controversy surrounding the book’s universalist views.{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}What is God?{tuple_delimiter}BOOK{tuple_delimiter}What is God? is the 1989 book that brought Etan Boritzer national critical acclaim and criticism from religious fundamentalists.{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}Religious Fundamentalists{tuple_delimiter}GROUP{tuple_delimiter}Religious fundamentalists are identified as critics of What is God? because of the book’s universalist views.{tuple_delimiter}75){record_delimiter}
("entity"{tuple_delimiter}Universalist Views{tuple_delimiter}IDEOLOGY{tuple_delimiter}Universalist views are the perspective in What is God? that caused controversy among religious fundamentalists.{tuple_delimiter}75){record_delimiter}
("entity"{tuple_delimiter}1989{tuple_delimiter}DATE{tuple_delimiter}1989 is the publication year linked to the acclaim and controversy surrounding What is God?.{tuple_delimiter}65){record_delimiter}
("hyper-relation"{tuple_delimiter}The current books in Etan Boritzer’s What is? series include What is Love?, What is Death?, What is Beautiful?, What is Funny?, What is Right?, What is Peace?, What is Money?, What is Dreaming?, What is a Friend?, What is True?, What is a Family?, and What is a Feeling?, and the series has been translated into 15 languages.{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}What is?{tuple_delimiter}BOOK_SERIES{tuple_delimiter}What is? is Etan Boritzer’s children’s book series that includes numerous question-titled books and has been translated into 15 languages.{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}What is Love?{tuple_delimiter}BOOK{tuple_delimiter}What is Love? is listed as one of the current books in Etan Boritzer’s What is? series.{tuple_delimiter}60){record_delimiter}
("entity"{tuple_delimiter}What is Death?{tuple_delimiter}BOOK{tuple_delimiter}What is Death? is listed as one of the current books in Etan Boritzer’s What is? series.{tuple_delimiter}60){record_delimiter}
("entity"{tuple_delimiter}What is Beautiful?{tuple_delimiter}BOOK{tuple_delimiter}What is Beautiful? is listed as one of the current books in Etan Boritzer’s What is? series.{tuple_delimiter}60){record_delimiter}
("entity"{tuple_delimiter}What is Funny?{tuple_delimiter}BOOK{tuple_delimiter}What is Funny? is listed as one of the current books in Etan Boritzer’s What is? series.{tuple_delimiter}60){record_delimiter}
("entity"{tuple_delimiter}What is Right?{tuple_delimiter}BOOK{tuple_delimiter}What is Right? is listed as one of the current books in Etan Boritzer’s What is? series.{tuple_delimiter}60){record_delimiter}
("entity"{tuple_delimiter}What is Peace?{tuple_delimiter}BOOK{tuple_delimiter}What is Peace? is listed as one of the current books in Etan Boritzer’s What is? series.{tuple_delimiter}60){record_delimiter}
("entity"{tuple_delimiter}What is Money?{tuple_delimiter}BOOK{tuple_delimiter}What is Money? is listed as one of the current books in Etan Boritzer’s What is? series.{tuple_delimiter}60){record_delimiter}
("entity"{tuple_delimiter}What is Dreaming?{tuple_delimiter}BOOK{tuple_delimiter}What is Dreaming? is listed as one of the current books in Etan Boritzer’s What is? series.{tuple_delimiter}60){record_delimiter}
("entity"{tuple_delimiter}What is a Friend?{tuple_delimiter}BOOK{tuple_delimiter}What is a Friend? is listed as one of the current books in Etan Boritzer’s What is? series.{tuple_delimiter}60){record_delimiter}
("entity"{tuple_delimiter}What is True?{tuple_delimiter}BOOK{tuple_delimiter}What is True? is listed as one of the current books in Etan Boritzer’s What is? series.{tuple_delimiter}60){record_delimiter}
("entity"{tuple_delimiter}What is a Family?{tuple_delimiter}BOOK{tuple_delimiter}What is a Family? is listed as one of the current books in Etan Boritzer’s What is? series.{tuple_delimiter}60){record_delimiter}
("entity"{tuple_delimiter}What is a Feeling?{tuple_delimiter}BOOK{tuple_delimiter}What is a Feeling? is listed as one of the current books in Etan Boritzer’s What is? series.{tuple_delimiter}60){record_delimiter}
("entity"{tuple_delimiter}15 Languages{tuple_delimiter}QUANTITY{tuple_delimiter}15 languages is the number of languages into which Etan Boritzer’s What is? series has been translated.{tuple_delimiter}75){record_delimiter}
("hyper-relation"{tuple_delimiter}Etan Boritzer was first published in 1963 at age 13, when an English-class essay he wrote at Wade Junior High School in the Bronx, New York, about the assassination of John F. Kennedy was included in a special anthology compiled and published by the New York City Department of Education.{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}Etan Boritzer{tuple_delimiter}PERSON{tuple_delimiter}Etan Boritzer was first published in 1963 at age 13 through an English-class essay about John F. Kennedy’s assassination.{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}1963{tuple_delimiter}DATE{tuple_delimiter}1963 is the year Etan Boritzer was first published at age 13.{tuple_delimiter}75){record_delimiter}
("entity"{tuple_delimiter}13{tuple_delimiter}AGE{tuple_delimiter}13 was Etan Boritzer’s age when his first published essay appeared in a special anthology.{tuple_delimiter}65){record_delimiter}
("entity"{tuple_delimiter}Wade Junior High School{tuple_delimiter}SCHOOL{tuple_delimiter}Wade Junior High School was the Bronx, New York school where Etan Boritzer wrote the English-class essay that became his first published work.{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}The Bronx, New York{tuple_delimiter}PLACE{tuple_delimiter}The Bronx, New York is the location of Wade Junior High School, where Etan Boritzer wrote his first published essay.{tuple_delimiter}80){record_delimiter}
("entity"{tuple_delimiter}John F. Kennedy{tuple_delimiter}PERSON{tuple_delimiter}John F. Kennedy was the subject of the assassination essay written by Etan Boritzer in 1963.{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}Assassination of John F. Kennedy{tuple_delimiter}EVENT{tuple_delimiter}The assassination of John F. Kennedy was the historical event addressed in Etan Boritzer’s English-class essay that was included in a special anthology.{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}New York City Department of Education{tuple_delimiter}ORGANIZATION{tuple_delimiter}The New York City Department of Education compiled and published the special anthology of New York City public school children’s writing that included Etan Boritzer’s essay.{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}New York City Public School Children{tuple_delimiter}GROUP{tuple_delimiter}New York City public school children were the contributors represented in the special anthology that included Etan Boritzer’s essay.{tuple_delimiter}65){record_delimiter}
("hyper-relation"{tuple_delimiter}Etan Boritzer now lives and maintains his publishing office in Venice, California; he has helped other authors get published through How to Get Your Book Published! programs, teaches yoga locally and nationally, and is nationally recognized as an erudite speaker on The Teachings of the Buddha.{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}Etan Boritzer{tuple_delimiter}PERSON{tuple_delimiter}Etan Boritzer currently lives and works in Venice, California, helps authors get published, teaches yoga locally and nationally, and speaks nationally on The Teachings of the Buddha.{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}Venice, California{tuple_delimiter}PLACE{tuple_delimiter}Venice, California is the place where Etan Boritzer currently lives and maintains his publishing office.{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}How to Get Your Book Published!{tuple_delimiter}PROGRAM{tuple_delimiter}How to Get Your Book Published! programs are initiatives through which Etan Boritzer has helped numerous authors get published.{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}Other Authors{tuple_delimiter}GROUP{tuple_delimiter}Other authors are the beneficiaries of Etan Boritzer’s publishing-support work through How to Get Your Book Published! programs.{tuple_delimiter}65){record_delimiter}
("entity"{tuple_delimiter}Yoga{tuple_delimiter}DISCIPLINE{tuple_delimiter}Yoga is a discipline Etan Boritzer teaches in regular local classes and as a guest teacher nationally.{tuple_delimiter}75){record_delimiter}
("entity"{tuple_delimiter}The Teachings of the Buddha{tuple_delimiter}TOPIC{tuple_delimiter}The Teachings of the Buddha is the topic on which Etan Boritzer is nationally recognized as an erudite speaker.{tuple_delimiter}85){record_delimiter}
#############################""",
]

PROMPTS["visual_entity_extraction"] = """-Purpose-
Extract structured high-order visual knowledge from raw images by identifying semantically coherent visual subscenes and their associated entities.

-Goal-
Given a raw image, its associated texts (caption, OCR, metadata), and a predefined list of entity types, identify visual knowledge segments and extract all visual entities associated with each segment.
Use {language} as output language.

-Input-
Image: {input_image}
Associated text: {associated_text}
Entity types: {entity_types}

-Steps-
1. Extract visual knowledge segments. Divide the image into the minimum number of self-contained, semantically coherent subscenes. Each segment must capture one coherent visual event, interaction, or functional cluster involving one or more entities simultaneously.
For each segment output:
-- visual_segment: A concise description of the subscene grounded in both image and associated text.
-- completeness_score: A score from 0 to 10 indicating how complete and self-contained the segment is.
Format each visual segment as ("hyper-relation"{tuple_delimiter}<visual_segment>{tuple_delimiter}<completeness_score>)

2. Extract visual entities. For each entity appearing in a segment, extract:
-- entity_name: Use the full explicit name as visible in the image or stated in associated text.
-- entity_type: Assign the correct entity type from the predefined list.
-- entity_description: Provide a context-rich description of the entity's role in that specific segment, including spatial, relational, and visual attribute details.
-- confidence_score: Assign a score from 0 to 100 reflecting the certainty in the extraction.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)

3. Return all extracted hyper-relations and entities as a single list. Use **{record_delimiter}** to separate records.

4. When finished, output {completion_delimiter}

#############################
-Real Data-
######################
Image: {input_image}
Associated text: {associated_text}
Entity types: {entity_types}
######################
Output:
"""

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
