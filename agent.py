from langchain.agents import initialize_agent, AgentType
from output_parser import get_output_parser


def setup_agent(llm, tools, memory, context_prompt):
    # Get Pydantic output parser
    output_parser = get_output_parser()
    
    # Initialize the agent 
    return initialize_agent(
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        memory=memory,
        verbose=True,
        output_parser=output_parser,  
    )
