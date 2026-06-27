"""
NERDC (Nigerian Educational Research and Development Council) Curriculum Loader
Structures primary school curriculum content for RAG ingestion.
Covers JSS 1–3 and Primary 1–6 core subjects.
Source: https://nerdc.gov.ng/
"""

from pathlib import Path
from typing import Iterator
from langchain.schema import Document


# ── Curriculum content database ───────────────────────────────────────────────
# Structured excerpt of NERDC curriculum — English medium, all 6 primary grades
# Each entry becomes a RAG document chunk.

CURRICULUM_CONTENT = [
    # ── MATHEMATICS ───────────────────────────────────────────────────────────
    {
        "subject": "Mathematics",
        "grade": 1,
        "topic": "Numbers 1–100",
        "content": (
            "Counting: Students learn to count from 1 to 100. "
            "They identify numbers, write them, and arrange them in order. "
            "Example: 1, 2, 3, ... 10, 11, ... 100. "
            "Skip counting by 2s: 2, 4, 6, 8, 10. By 5s: 5, 10, 15, 20. By 10s: 10, 20, 30."
        ),
    },
    {
        "subject": "Mathematics",
        "grade": 1,
        "topic": "Addition and Subtraction (within 20)",
        "content": (
            "Addition means joining numbers together. The + sign is used. "
            "Example: 5 + 3 = 8. Subtraction means taking away. The − sign is used. "
            "Example: 9 − 4 = 5. "
            "Real-life context: If you have 6 naira and spend 2 naira, you have 4 naira left."
        ),
    },
    {
        "subject": "Mathematics",
        "grade": 2,
        "topic": "Multiplication Tables (2–5)",
        "content": (
            "Multiplication is repeated addition. "
            "2 × 3 means 2 + 2 + 2 = 6. "
            "Table of 2: 2×1=2, 2×2=4, 2×3=6, 2×4=8, 2×5=10. "
            "Table of 3: 3×1=3, 3×2=6, 3×3=9, 3×4=12, 3×5=15. "
            "Real life: If 1 bag of rice costs ₦500, 3 bags cost ₦1500."
        ),
    },
    {
        "subject": "Mathematics",
        "grade": 3,
        "topic": "Fractions",
        "content": (
            "A fraction is part of a whole. Written as numerator/denominator. "
            "1/2 = one half (divide into 2 equal parts, take 1). "
            "1/4 = one quarter. 3/4 = three quarters. "
            "Example: A piece of cloth 1 metre long, cut into 4 equal pieces. "
            "Each piece is 1/4 metre. Three pieces = 3/4 metre. "
            "Equivalent fractions: 1/2 = 2/4 = 4/8."
        ),
    },
    {
        "subject": "Mathematics",
        "grade": 4,
        "topic": "Long Division",
        "content": (
            "Long division breaks a large number into groups. "
            "Steps: Divide, Multiply, Subtract, Bring down (DMSB). "
            "Example: 156 ÷ 4. "
            "4 goes into 15 three times (3×4=12). Remainder 3. Bring down 6 → 36. "
            "4 goes into 36 nine times (9×4=36). Answer: 39. "
            "Check: 39 × 4 = 156 ✓"
        ),
    },
    {
        "subject": "Mathematics",
        "grade": 5,
        "topic": "Percentages",
        "content": (
            "Percent means out of 100. The symbol is %. "
            "50% = 50/100 = 1/2. 25% = 25/100 = 1/4. "
            "To find 20% of 500: 20/100 × 500 = 100. "
            "Application: A trader buys goods for ₦2,000 and sells for ₦2,400. "
            "Profit = ₦400. Profit% = (400/2000) × 100 = 20%."
        ),
    },
    {
        "subject": "Mathematics",
        "grade": 6,
        "topic": "Simple Interest and Business Mathematics",
        "content": (
            "Simple Interest = (Principal × Rate × Time) / 100. "
            "P = amount borrowed/saved. R = interest rate per year. T = time in years. "
            "Example: Amina saves ₦10,000 in a bank at 5% per year for 3 years. "
            "Interest = (10000 × 5 × 3)/100 = ₦1,500. Total = ₦11,500. "
            "Real life: Microfinance loans to farmers use simple interest."
        ),
    },
    # ── ENGLISH LANGUAGE ──────────────────────────────────────────────────────
    {
        "subject": "English Language",
        "grade": 1,
        "topic": "Phonics and Alphabets",
        "content": (
            "The English alphabet has 26 letters: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z. "
            "Vowels: A, E, I, O, U. All other letters are consonants. "
            "Each letter has a sound. B says /b/ as in 'ball'. C says /k/ as in 'cat'. "
            "Blending: b-a-t = bat. c-a-t = cat. d-o-g = dog."
        ),
    },
    {
        "subject": "English Language",
        "grade": 2,
        "topic": "Nouns and Pronouns",
        "content": (
            "A noun is a name word — a word that names a person, place, animal, or thing. "
            "Person: teacher, doctor, child, farmer. Place: school, market, Lagos, river. "
            "Animal: goat, dog, eagle, fish. Thing: book, pen, water, food. "
            "A pronoun replaces a noun: I, you, he, she, it, we, they. "
            "Example: 'Chidi reads.' → 'He reads.' 'Ngozi cooks.' → 'She cooks.'"
        ),
    },
    {
        "subject": "English Language",
        "grade": 3,
        "topic": "Letter Writing",
        "content": (
            "A formal letter has five parts: "
            "1. Your address (top right) 2. Date 3. Receiver's address (left) "
            "4. Salutation: Dear Sir/Madam 5. Body — state your purpose clearly. "
            "6. Closing: Yours faithfully (formal) / Yours sincerely (informal). "
            "Signature and name. "
            "Informal letters use Dear [name] and Yours lovingly / Your friend."
        ),
    },
    {
        "subject": "English Language",
        "grade": 4,
        "topic": "Comprehension and Summary",
        "content": (
            "Comprehension means understanding a passage. "
            "Steps: Read the passage carefully (twice if needed). "
            "Read the questions. Find the answers in the passage. "
            "Write answers in complete sentences using your own words. "
            "Summary: Pick only the main points. Remove examples and repetitions. "
            "Keep it short — usually 1/3 of the original length."
        ),
    },
    # ── BASIC SCIENCE ─────────────────────────────────────────────────────────
    {
        "subject": "Basic Science",
        "grade": 2,
        "topic": "States of Matter",
        "content": (
            "Matter is anything that has weight and takes up space. Three states: "
            "1. Solid: has a definite shape and size. Examples: stone, wood, ice, book. "
            "2. Liquid: has no fixed shape; takes the shape of its container. Examples: water, palm oil, milk. "
            "3. Gas: has no fixed shape or size; spreads to fill any space. Examples: air, steam, smoke. "
            "Change of state: Water → heated → Steam (gas). Water → cooled → Ice (solid)."
        ),
    },
    {
        "subject": "Basic Science",
        "grade": 3,
        "topic": "The Water Cycle",
        "content": (
            "The water cycle is the continuous movement of water in nature. "
            "4 stages: "
            "1. Evaporation: Sun heats water in rivers, lakes, and oceans. Water turns to water vapour (invisible gas) and rises. "
            "2. Condensation: High in the sky, water vapour cools and turns into tiny water droplets forming clouds. "
            "3. Precipitation: Water falls from clouds as rain, hail, or snow. "
            "4. Collection: Water collects in rivers, lakes, and underground. The cycle repeats. "
            "Importance: Provides fresh water for drinking, farming, and animals."
        ),
    },
    {
        "subject": "Basic Science",
        "grade": 4,
        "topic": "Photosynthesis",
        "content": (
            "Photosynthesis is how plants make their own food using sunlight. "
            "Formula: Carbon dioxide + Water + Sunlight → Glucose + Oxygen. "
            "Where it happens: In the green parts of the plant, especially leaves. "
            "Chlorophyll (the green colour) captures sunlight. "
            "Importance: Plants release oxygen that humans and animals breathe. "
            "Plants are the base of all food chains. Without photosynthesis, there would be no food on Earth."
        ),
    },
    # ── SOCIAL STUDIES ────────────────────────────────────────────────────────
    {
        "subject": "Social Studies",
        "grade": 3,
        "topic": "Nigeria — Our Country",
        "content": (
            "Nigeria is a country in West Africa. It became independent on October 1, 1960. "
            "Capital city: Abuja. Largest city: Lagos. "
            "Nigeria has 36 states and the Federal Capital Territory (FCT). "
            "Six geopolitical zones: North-West, North-East, North-Central, South-West, South-East, South-South. "
            "Population: Over 220 million people — largest in Africa. "
            "Main languages: Hausa, Yoruba, Igbo, with over 500 other languages. "
            "National anthem: 'Arise O Compatriots'. National motto: Unity and Faith, Peace and Progress."
        ),
    },
    {
        "subject": "Social Studies",
        "grade": 5,
        "topic": "Democracy and Citizenship",
        "content": (
            "Democracy is a system of government where the people choose their leaders by voting. "
            "Key principles: Freedom, Equality, Justice, Rule of Law. "
            "In Nigeria, elections are held every 4 years for President, Governors, and lawmakers. "
            "INEC (Independent National Electoral Commission) organises elections. "
            "Rights of citizens: Right to vote (from age 18), right to education, right to free speech. "
            "Duties: Pay taxes, obey the law, vote, serve on juries, respect others."
        ),
    },
    # ── CIVIC EDUCATION ──────────────────────────────────────────────────────
    {
        "subject": "Civic Education",
        "grade": 4,
        "topic": "National Values",
        "content": (
            "National values are beliefs that guide how Nigerians should live together. "
            "Discipline: doing what is right even when nobody is watching. "
            "Integrity: being honest and trustworthy. "
            "Dignity of labour: all work is honourable — farming, teaching, trading. "
            "Social justice: treating everyone fairly regardless of tribe or religion. "
            "Religious tolerance: respecting other people's right to their religion. "
            "These values make Nigeria peaceful and united."
        ),
    },
    # ── HEALTH & PHYSICAL EDUCATION ──────────────────────────────────────────
    {
        "subject": "Health & Physical Education",
        "grade": 2,
        "topic": "Personal Hygiene",
        "content": (
            "Hygiene means keeping your body clean to stay healthy. "
            "Daily habits: Brush teeth morning and night (2 minutes). Bathe every day with soap. "
            "Wash hands with soap before eating and after using the toilet. "
            "Cut fingernails short — dirty nails carry germs. "
            "Wash clothes regularly. Keep hair clean and combed. "
            "Why hygiene matters: Prevents diseases like cholera, typhoid, and diarrhoea. "
            "Clean children learn better and miss fewer school days."
        ),
    },
    # ── AGRICULTURAL SCIENCE ─────────────────────────────────────────────────
    {
        "subject": "Agricultural Science",
        "grade": 5,
        "topic": "Crop Production",
        "content": (
            "Crops are plants grown by farmers for food or other uses. "
            "Food crops: Maize, yam, cassava, rice, groundnut, soybean. "
            "Cash crops: Cocoa, cotton, palm oil, rubber. "
            "Steps in crop production: "
            "1. Land preparation (clearing, tilling) 2. Planting (seeds or seedlings) "
            "3. Fertilising (NPK or compost) 4. Weeding 5. Pest control 6. Harvesting. "
            "Nigeria's main farming zones: Guinea savanna (cereals), forest belt (cassava, cocoa)."
        ),
    },
]


def load_curriculum_documents(
    subject: str | None = None,
    grade: int | None = None,
) -> list[Document]:
    """Return LangChain Documents filtered by subject and/or grade."""
    docs = []
    for item in CURRICULUM_CONTENT:
        if subject and item["subject"].lower() != subject.lower():
            continue
        if grade and item["grade"] != grade:
            continue
        metadata = {
            "source": f"NERDC National Curriculum — {item['subject']}, Primary {item['grade']}",
            "subject": item["subject"],
            "grade": item["grade"],
            "topic": item["topic"],
        }
        docs.append(Document(page_content=item["content"], metadata=metadata))
    return docs


def load_all_documents() -> list[Document]:
    return load_curriculum_documents()
