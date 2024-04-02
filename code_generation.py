import json
import os
import random
import textwrap
from enum import Enum
from inspect import cleandoc
from pathlib import Path
from typing import Literal, Optional

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from tenacity import retry, stop_after_attempt, wait_random_exponential

load_dotenv()

ask_oai_async = instructor.patch(AsyncOpenAI()).chat.completions.create
ask_oai = instructor.patch(OpenAI()).chat.completions.create

TEMPERATURE = 0.3
ATTEMPTS = 2

NUM_TOPICS = 10
NUM_SUBTOPICS = 5

RETRY_ARGS = dict(
    wait=wait_random_exponential(min=60, max=120), stop=stop_after_attempt(6)
)


def deindent(text: str) -> str:
    return textwrap.dedent(cleandoc(text))


def chat_message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def system_message(content: str) -> dict[str, str]:
    return chat_message(role="system", content=content)


def user_message(content: str) -> dict[str, str]:
    return chat_message(role="user", content=content)


class ModelName(str, Enum):
    GPT_3 = "gpt-3.5-turbo"
    GPT_4 = "gpt-4-turbo-preview"


class LearningObjective(str, Enum):
    KNOWLEDGE = "knowledge"
    COMPREHENSION = "comprehension"
    APPLICATION = "application"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"


class Topic(BaseModel):
    name: str = Field(..., title="Name of topic")
    sub_topics: list[str] = Field(
        ...,
        title="List of subtopics",
        min_length=NUM_SUBTOPICS - 1,
        max_length=NUM_SUBTOPICS,
    )


class Topics(BaseModel):
    topics: list[Topic] = Field(
        ..., title="List of topics", min_length=NUM_TOPICS - 1, max_length=NUM_TOPICS
    )


class QuestionMeta(BaseModel):
    topic_name: str = Field(..., title="Name of topic")
    subtopic_name: str = Field(..., title="Name of subtopic")
    learning_objectives: list[LearningObjective] = Field(
        ..., title="List of learning objectives"
    )

    @field_validator("topic_name")
    @classmethod
    def validate_topic_name(cls, topic_name: str, info: ValidationInfo) -> str:
        topic_names = info.context.get("topic_names", [])
        if topic_name not in topic_names:
            raise ValueError(f"{topic_name} is not a valid topic")
        return topic_name

    @model_validator(mode="after")
    def validate_subtopic_name(self, info: ValidationInfo) -> "QuestionMeta":
        topics_dict = info.context.get("topics_dict", {})
        sub_topics = topics_dict.get(self.topic_name, [])
        if self.subtopic_name in sub_topics:
            return self
        raise ValueError(
            f"{self.subtopic_name} is not a valid subtopic for {self.topic_name}"
        )


class MathCodeGen(BaseModel):
    problem: str = Field(
        ...,
        description="A problem specified in words or as mathematical equations and functions",
    )
    solution: str = Field(
        ..., description="A step by step solution to the given problem"
    )
    prev_code: Optional[str] = Field(
        "", description="A previously generated Python function code"
    )
    feedback: Optional[str] = Field(
        "", description="User feedback on the previous Python function"
    )
    reasoning: str = Field(
        ..., description="Step by step reasoning for generating the Python code"
    )
    python_code: str = Field(
        ...,
        description=deindent(
            """
            A Python function that re-generates the question with random values and converts its solution steps into sub-questions.
            Your response should start with ```python and end with ```
            """
        ),
        pattern=r"^```python[\s\S]*```$",
    )


@retry(**RETRY_ARGS)
async def classify_question(
    question_id: int,
    question: dict,
    classes: list[str],
    model: ModelName = ModelName.GPT_3,
    attempts: int = ATTEMPTS,
    folder: str = "questions_with_topics",
    class_name: str = "topic",
) -> str:
    os.makedirs(folder, exist_ok=True)
    current_ids = [x.stem for x in Path(folder).glob("*.json")]
    if str(question_id) in current_ids:
        res = json.load(open(f"{folder}/{question_id}.json"))[class_name]
        return res
    messages = [
        system_message("You are a world class course instructor."),
        user_message(
            f"Classify the following question into one of the following classes: {classes}\nQUESTION:\n{question}"
        ),
    ]
    res = await ask_oai_async(
        messages=messages,
        model=model,
        response_model=Literal[tuple(classes)],  # type: ignore
        max_retries=attempts,
        temperature=TEMPERATURE,
    )
    with open(f"{folder}/{question_id}.json", "w") as f:
        json.dump({**question, class_name: res}, f, indent=2)
    return res


@retry(**RETRY_ARGS)
async def assign_slo(
    question_id: int,
    question: dict,
    model: ModelName = ModelName.GPT_3,
    attempts: int = ATTEMPTS,
    folder: str = "questions_with_slos",
) -> list[LearningObjective]:
    os.makedirs(folder, exist_ok=True)
    current_ids = [x.stem for x in Path(folder).glob("*.json")]
    if str(question_id) in current_ids:
        res = json.load(open(f"{folder}/{question_id}.json"))["learning_objectives"]
        return res
    slos = [x.lower() for x in list(LearningObjective.__members__.keys())]
    messages = [
        system_message("You are a world class course instructor."),
        user_message(
            f"Classify the following question into one or or more of the following learning objectives: {slos}\nTEXT:\n{question}"
        ),
    ]
    res = await ask_oai_async(
        messages=messages,
        model=model,
        response_model=list[LearningObjective],
        max_retries=attempts,
        temperature=TEMPERATURE,
    )
    with open(f"{folder}/{question_id}.json", "w") as f:
        json.dump({**question, "learning_objectives": res}, f, indent=2)
    return res


def query_to_questions(
    query: str,
    course_name: str,
    model: ModelName = ModelName.GPT_3,
    n_questions: int = 5,
    attempts: int = ATTEMPTS,
) -> list[dict]:
    topics_dict = json.load(open(f"{course_name}_topics.json"))
    topics = Topics(**topics_dict)
    topics_dict = {t.name: t.sub_topics for t in topics.topics}
    topics_list = [
        {"topic": topic, "subtopics": subtopics}
        for topic, subtopics in topics_dict.items()
    ]
    topic_names = list(topics_dict.keys())
    questions = json.load(open(f"{course_name}_questions.json"))

    task = system_message(
        deindent(f"""
                 You are a world class course instrcutor. The user will be looking for a certain type of question.
                 Your job is to extract the topic, subtopic, and learning objectives from the user's request.
                 That will help us select the right questions for the user.
                 Here are the topics and subtopics you can choose from:
                    {topics_list}
                 """)
    )
    messages = [task, user_message(query)]
    res = ask_oai(
        messages=messages,
        model=model,
        response_model=QuestionMeta,
        max_retries=attempts,
        temperature=TEMPERATURE,
        validation_context={"topic_names": topic_names, "topics_dict": topics_dict},
    )
    topic_qs = [q for q in questions if q["topic"] == res.topic_name]
    subtopic_qs = [q for q in topic_qs if q["sub_topic"] == res.subtopic_name]
    slo_qs = [
        q
        for q in subtopic_qs
        if any([lo in q["learning_objectives"] for lo in res.learning_objectives])
    ]
    if len(slo_qs) < n_questions:
        return slo_qs
    return random.sample(slo_qs, n_questions)


def generate_code(
    question: dict,
    prev_code: str = "",
    feedback: str = "",
    model: ModelName = ModelName.GPT_4,
    prompt_path: str = "code_prompt.txt",
    attempts: int = ATTEMPTS,
) -> str:
    prompt = f"Problem: {question['problem']}\n\nSolution: {question['solution']}"
    if prev_code:
        prompt += f"\n\nPrevious Code: {prev_code}"
    if feedback:
        prompt += f"\n\nFeedback: {feedback}"
    code_messages = [
        system_message(Path(prompt_path).read_text()),
        user_message(prompt),
    ]
    code = ask_oai(
        messages=code_messages,
        model=model,
        response_model=MathCodeGen,
        max_retries=attempts,
        temperature=TEMPERATURE,
    )
    return code.python_code
