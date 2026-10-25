
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama


# Tool 1: Calculate square
@tool
def calculate_square(number: int) -> int:
    """Calculate the square of a number.
    Use this tool when the user asks to calculate a square.
    """
    return number * number


# Tool 2: Calculate sum
@tool
def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers.
    Use this tool when the user asks to add two numbers.
    """
    return a + b


# Tool 3: Check basic loan eligibility
@tool
def check_loan_eligibility(monthly_income: float,credit_score: int) -> str:

    """Demonstrate a basic loan eligibility check using income and credit score.
    Use this tool when the user asks about loan eligibility.
    This is only a simplified demonstration, not a real bank decision.
    """

    if monthly_income <= 0:
        return "Monthly income must be greater than zero."

    if not 300 <= credit_score <= 900:
        return "Please provide a credit score between 300 and 900."

    if monthly_income >= 25000 and credit_score >= 700:
        return (
            "The applicant meets this demo's basic criteria. "
            "Actual approval depends on the bank's policies and "
            "a complete financial assessment."
        )

    return (
        "The applicant does not meet this demo's basic criteria. "
        "Actual eligibility depends on the bank's policies and "
        "a complete financial assessment."
    )


# Tool 4: Banking support
@tool
def get_banking_support(topic: str) -> str:
    """Get general banking support information.

    Call this tool when the user asks about:
    - Online banking or internet banking
    - Savings accounts
    - Credit cards
    - Banking customer support

    Args:
        topic: The banking topic the user needs help with.
    """
    topic = topic.lower()

    if "online" in topic or "internet" in topic:
        return (
            "For online banking support, visit your bank's "
            "official website or application."
        )

    elif "credit card" in topic:
        return "Check your credit card balance and payment due date."

    elif "account" in topic or "savings" in topic:
        return "Contact your bank for savings account support."

    return "Contact your bank's customer support team."



def create_banking_agent():

    # Connect to the local Ollama model
    model = ChatOllama(
        model="qwen2.5:0.5b",
        temperature=0,
        base_url="http://127.0.0.1:11434",
    )

    # Register tools with the agent
    tools = [
        calculate_square,
        calculate_sum,
        check_loan_eligibility,
        get_banking_support,
    ]

    # Create the agent
    # agent = create_agent(
    #     model=model,
    #     tools=tools,
    #     system_prompt=(
    #         "You are a helpful banking assistant. "
    #         "Use the available tools whenever they are relevant. "
    #         "For mathematical questions, use the appropriate "
    #         "calculation tool. "
    #         "For loan eligibility questions, use the loan tool. "
    #         "For general banking questions, use the banking support tool. "
    #         "Explain results in simple English. "
    #         "Never claim that demo loan results guarantee real approval."
    #     ),
    # )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are a banking assistant.\n"
            "Rules:\n"
            "1. For square calculations, use calculate_square.\n"
            "2. For addition, use calculate_sum.\n"
            "3. For loan eligibility, use check_loan_eligibility.\n"
            "4. For banking support, use get_banking_support.\n"
            "5. When a registered tool is relevant, call it rather "
            "than answering from memory."
        ),
    )

    return agent


def ask_agent(question: str):

    agent = create_banking_agent()
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ]
    })

    return result["messages"][-1].content
