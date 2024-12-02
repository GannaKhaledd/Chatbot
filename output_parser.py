from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser

# Define the structure of the agent's output
class AgentOutput(BaseModel):
    thought: str
    final_answer: str

# Custom Pydantic output parser
def get_output_parser():
    return PydanticOutputParser(
        pydantic_object=AgentOutput,
        exception_handler=lambda e: f"Parsing error encountered: {str(e)}",
    )
