from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from datetime import datetime
import logging
import uvicorn

# Load environment variables
load_dotenv()

# Set API Keys from .env file
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM (Groq via OpenAI-compatible endpoint)
llm = None
if GROQ_API_KEY:
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        openai_api_key=GROQ_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0
    )

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the search tool
tool = TavilySearchResults(max_results=3) if TAVILY_API_KEY else None

# Define the AgentState data structure
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# Base Agent class
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")   # Loop Back
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    # Processes inputs using the model.
    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}   # Updates the state with the model's response

    # Checks if tools need to be invoked
    def exists_action(self, state: AgentState):
        result = state['messages'][-1]     # Checks the latest message in the state
        return len(result.tool_calls) > 0        # If the model’s output includes tool calls, return True

    # Executes tool calls and retrieves results
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            logging.info(f"Calling tool: {t['name']} with query: {t['args']['query']}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        logging.info("Returning results to the model...")
        return {'messages': results}

# Research the Industry or the Company Agent
research_prompt = """
You are a market research expert specializing in industry analysis and competitive insights. Use available tools, including web-based research, to conduct a thorough and in-depth analysis of the {query_context}.

Key goals:
1. **Understand the Industry and Segment**: Provide a detailed overview of the industry or segment the company is operating in (e.g., Automotive, Manufacturing, Finance, Retail, Healthcare, etc.). Include insights on market trends, segmentation, and challenges.

2. **Analyze the Company's Offerings and Focus Areas**:
   - Identify the company’s key offerings and their strategic focus areas (e.g., operations, supply chain, customer experience, etc.).
   - Summarize the company’s vision, mission, and product/service portfolio.

3. **Competitor Analysis**:
   - Provide a detailed list of competitors in the same industry or segment.
   - Highlight examples of their AI, GenAI, and ML applications or innovations.
   - Assess their strategies and identify differentiators that the company can leverage.

4. **Quantitative Data**: Include relevant metrics, such as market size, growth rates, technology adoption percentages, and revenue impacts, to support your findings.

5. **Actionable Insights**: Summarize clear and actionable insights that:
   - Highlight opportunities for AI/ML/automation in the company’s operations.
   - Guide further exploration and decision-making for generating impactful AI use cases.

6. **Depth of Analysis**: Ensure the research demonstrates both breadth and depth, providing a comprehensive understanding of the market landscape, the competitive environment, and technological adoption trends.

Output: Deliver a detailed yet concise analysis with insights that help generate tailored AI use cases.
"""

research_agent = Agent(llm, [tool] if tool else [], system=research_prompt) if llm else None

# Market Standards & Use Case Generation Agent
use_case_prompt = """
You are an expert in AI use case generation, specializing in industry-specific innovation. Based on the following industry research, generate actionable and impactful AI/GenAI use cases strictly related to AI, Machine Learning (ML), and Automation, tailored to the industry’s pain points and opportunities.

Key goals:
1. Focus exclusively on use cases that leverage:
   - Artificial Intelligence (AI)
   - Machine Learning (ML)
   - Generative AI (GenAI)
   - Automation technologies
   - Large Language Models (LLMs)

2. Ensure all use cases are:
   - **Relevant**: Directly address the key challenges, opportunities, and goals specific to the industry or company.
   - **Creative**: Introduce innovative and forward-thinking applications of AI/ML/automation that go beyond conventional solutions.

3. Use a clear and structured format for each use case:
   - **Objective/Use Case**: Describe the primary goal and the specific application area.
   - **AI/ML/Automation Application**: Clearly explain how AI, ML, or automation is utilized to address the use case.
   - **Cross-Functional Benefits**: Highlight the benefits across multiple departments or functions, using bullets or subheadings.

4. Provide a list of at least five impactful use cases, formatted as follows:
   ### AI, ML & Automation Use Cases for [Industry or Company Name]

   As a leading player in [Industry/Company context], [Company/Industry Name] can leverage Artificial Intelligence (AI), Machine Learning (ML), Generative AI (GenAI), and Automation to [overall benefit statement]. The following use cases can be realized:

   **Use Case 1: [Use Case Title]**
   - **Objective/Use Case**: [Description]
   - **AI/ML/Automation Application**: [Description of how AI, ML, or automation is applied.]
   - **Cross-Functional Benefit**:
     - [Department 1]: [Benefit]
     - [Department 2]: [Benefit]

   **Use Case 2: [Use Case Title]**
   - **Objective/Use Case**: [Description]
   - **AI/ML/Automation Application**: [Description of how AI, ML, or automation is applied.]
   - **Cross-Functional Benefit**:
     - [Department 1]: [Benefit]
     - [Department 2]: [Benefit]

   Repeat the above format for all proposed use cases.

5. Summarize actionable insights clearly to guide stakeholders in prioritization and implementation.

**Remember**: Relevance and creativity are critical in generating use cases that demonstrate the unique value AI/ML/automation can bring to the industry or company.
"""

use_case_agent = Agent(llm, [tool] if tool else [], system=use_case_prompt) if llm else None

# Resource Asset Collection Agent
resource_collection_prompt = """
You are an expert in resource asset collection, tasked with identifying datasets, tools, frameworks, and proposing actionable Generative AI (GenAI) solutions.

Key goals:
1. Search for relevant datasets on platforms like Kaggle, HuggingFace, GitHub, and others to support the proposed use cases.
2. Identify pre-trained models, APIs, or open-source tools that align with each use case.
3. Propose Generative AI (GenAI) solutions, such as:
   - **Document Search**: AI-powered tools for semantic search of internal or external documents.
   - **Automated Report Generation**: Tools or frameworks for generating tailored reports based on provided inputs.
   - **AI-Powered Chat Systems**: Virtual assistants for customer support or operational tasks.
4. Ensure that all resources and solutions are practical, accessible, and include clickable links or examples for easy implementation.

Output: Deliver a detailed list of datasets, tools, resources, and GenAI solutions organized by use case. Provide clear links and descriptions for each.
"""

resource_agent = Agent(llm, [tool] if tool else [], system=resource_collection_prompt) if llm else None

def build_fallback_content(query: str):
    topic = query.strip() or "your industry"
    research = f"""Market research overview for {topic}

- Market context: {topic} is a fast-moving area where operational efficiency, customer experience, and automation are key value drivers.
- Market trends: Demand is shifting toward more personalized offerings, lower-friction operations, and faster decision-making.
- Recommended focus: Prioritize high-volume workflows, decision support, and workflow automation where AI can create measurable ROI.

Suggested next step: Add a valid GROQ_API_KEY to switch from this fallback to live Groq-generated research.
"""

    use_cases = f"""AI / ML use cases for {topic}

1. Intelligent forecasting and planning
   - Use AI to predict demand, inventory, and staffing needs more accurately.

2. Workflow automation and document processing
   - Automate repetitive document review, triage, and internal knowledge retrieval.

3. Customer support augmentation
   - Combine LLMs with internal knowledge to answer common support questions faster.

4. Recommendation and personalization
   - Tailor offers, content, and user journeys using historical behavior data.

5. Risk monitoring and anomaly detection
   - Detect unusual patterns in transactions, operations, or customer behavior.
"""

    resources = f"""Starter resources for {topic}

- Kaggle datasets and public repositories related to {topic}
- Open-source tools such as LangChain, LangGraph, and Gradio for rapid prototyping
- Groq-compatible LLM endpoints for production-ready inference
- Internal documentation, CSV exports, and CRM data to build a first proof of concept

To activate live Groq generation, add GROQ_API_KEY to your environment or .env file.
"""

    return research, use_cases, resources


# Function to save resources to a file
def save_resources_to_file(content: str, directory: str = "output"):
    """
    Save the resource content into a uniquely named file based on the current timestamp.

    Args:
        content (str): The resource content to save.
        directory (str): The directory where the file will be saved.

    Returns:
        str: The path of the saved file.
    """
    try:
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resources_{timestamp}.md"
        file_path = os.path.join(directory, filename)

        # Write content to the file
        with open(file_path, "w") as file:
            file.write(content)

        logging.info(f"Resources saved successfully to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Failed to save file: {e}")
        raise


# Multi-Agent Workflow
def multi_agent_workflow(question):
    if not llm or not research_agent or not use_case_agent or not resource_agent:
        logging.warning("Groq is not configured. Using local fallback content.")
        research_content, use_case_content, resource_content = build_fallback_content(question)
        file_path = save_resources_to_file(resource_content)
        logging.info(f"Fallback resource file saved at: {file_path}")
        return research_content, use_case_content, resource_content

    # Step 1: Research the Industry or the Company
    research_messages = [HumanMessage(content=question)]
    research_state = {"messages": research_messages}
    research_result = research_agent.graph.invoke(research_state)
    research_content = research_result["messages"][-1].content

    # Step 2: Generate Use Cases
    use_case_messages = [
        HumanMessage(content=f"Input Research: {research_content}")
    ]
    use_case_state = {"messages": use_case_messages}
    use_case_result = use_case_agent.graph.invoke(use_case_state)
    use_case_content = use_case_result["messages"][-1].content

    # Step 3: Collect Resources
    resource_messages = [
        HumanMessage(content=f"Use Case Input: {use_case_content}")
    ]
    resource_state = {"messages": resource_messages}
    resource_result = resource_agent.graph.invoke(resource_state)
    resource_content = resource_result["messages"][-1].content

    # Save resource content with a unique file name
    file_path = save_resources_to_file(resource_content)
    logging.info(f"Resource file saved at: {file_path}")

    return research_content, use_case_content, resource_content


app = FastAPI(title="AI Market Research API")


@app.get("/")
def index():
    return FileResponse("index.html")


class ResearchRequest(BaseModel):
    query: str


@app.post("/api/research")
def research_api(payload: ResearchRequest):
    try:
        if not payload.query or not payload.query.strip():
            raise HTTPException(status_code=400, detail="Please enter a query")

        research_content, use_case_content, resource_content = multi_agent_workflow(payload.query.strip())
        model_name = "groq/llama-3.3-70b-versatile" if llm else "local-fallback"
        return {
            "research": research_content,
            "useCases": use_case_content,
            "resources": resource_content,
            "model": model_name
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
