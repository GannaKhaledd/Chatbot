import os  
import json
import shutil
import panel as pn
from groq import Groq 
from langchain.agents import Tool
from langchain_groq import ChatGroq
from langchain.schema import Document  
from dotenv import load_dotenv, find_dotenv  
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma    
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import CSVLoader  
from langchain_core.exceptions import OutputParserException
from langchain.agents import  AgentExecutor, create_react_agent
from langchain_community.embeddings import SentenceTransformerEmbeddings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
load_dotenv(find_dotenv())
api_key = os.environ.get("GROQ_API_KEY")  

client = Groq(api_key=api_key)
llm_model  = "llama3-groq-70b-8192-tool-use-preview"
llm = ChatGroq(temperature=0.0, model=llm_model)

persist_directory = 'doc/chroma/'
if os.path.exists(persist_directory): 
    shutil.rmtree(persist_directory) 

file = 'products.csv'
loader = CSVLoader(file_path=file, encoding='utf-8')  
data = loader.load()  

def create_documents(data):
    documents = []
    for doc in data:
        lines = doc.page_content.split('\n')
        content_dict = {}
        for line in lines:
            key, value = line.split(": ")
            content_dict[key.strip()] = value.strip()
        content = f"Category: {content_dict['Category']}\nProduct: {content_dict['Product']}\nPrice: {content_dict['Price']}\nDescription: {content_dict['Description']}"
        documents.append(Document(page_content=content))
    return documents

documents = create_documents(data)

embedding = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#----------------------------------------------------------------------------------------------
cart = []
def add_to_cart (product_name):
    """Add a specified product to the cart."""
    for product in data:
        if product ['Product'].lower() == product_name.lower():
            cart.append(product)
            return product['Product'] + "has been added to your cart"
    return "The product you are trying to add to your cart is not available write now."
#----------------------------------------------------------------------------------------------
def make_an_order():
    """Generate an order based on items in the cart."""
    if len (cart) == 0  :
        return "Your cart is empty, please fill in you cart and comeback again"
    order = " "
    for item in cart :
        order += item ['Product'] + ", "
        order = order.rstrip(", ") 
    return "Your order is " + order
#----------------------------------------------------------------------------------------------
def calculate_total_price():
    """Calculate the total price of items in the cart."""
    total_price = 0
    for item in cart:
        total_price += item['Price']
    return total_price
#----------------------------------------------------------------------------------------------
def do_payment():
    """Process payment for the order."""
    if len (cart) == 0  :
        return "Your cart is empty, please fill in you cart and comeback again"
    order = make_an_order() 
    if "Your order is" not in order:
        return order  
    payment_method = input("How would you like to pay? (Cash/Credit card):").strip().lower()
    if payment_method == 'Cash':
        address = input(" Please provide your address for the delivery : ")
        total_price = calculate_total_price()  
        return f"Your order price is {total_price}. It will be delivered to {address}. Thank you for your order! Goodbye!"
    elif payment_method == 'Credit card':
        credit_details = input("Please enter your Credit card details: ")
        payment_successful = True 
        if payment_successful:
            return f"Payment was successful! for {order} Thank you for your order!"
        else:
            return "There was an issue with your Visa payment. Please try again."
    else:
        return "Invalid payment method selected. Please choose 'Cash' or 'Credit card'."
#----------------------------------------------------------------------------------------------
tools = [
    Tool(
        name="Add to Cart",
        func=add_to_cart,
        description="Add a specified product to the cart."
    ),
    Tool(
        name="Make an Order",
        func=make_an_order,
        description="Generate an order based on items in the cart."
    ),
    Tool(
        name="Calculate Total Price",
        func=calculate_total_price,
        description="Calculate the total price of items in the cart."
    ),
    Tool(
        name="Do Payment",
        func=do_payment,
        description="Process payment for the order."
    ),
]
tool_names = ['Add to Cart', 'Make an Order','Calculate Total Price', 'Do Payment']
#----------------------------------------------------------------------------------------------
context = """
You are a friendly bot designed to assist customers with questions about an electronic store.
Your goal is to provide helpful and concise information while maintaining a conversational tone.
Instructions:
1. **Greeting**: Start by greeting the customer in a friendly manner.
2. **Personal info**: When the user shares their name, acknowledge it warmly and engage them personally.
3. **Out of Context**: If a question is unrelated to the products or the store, simply state that you don't have information on that topic while maintaining a friendly tone. 
4. **Handling Harmful Content**: If questions contain harmful or inappropriate content, politely inform the user that you cannot assist with such inquiries.
5. **Avoid repeating Apologies**: There's no need to start every response with an apology. Be friendly and direct instead.
6. **Focus on Electronics**: Keep the conversation centered around the electronic products available and do not suggest unrelated topics.
7. **Product availability**: If the customer asks for a product that is not available, handle this in a kind way.
8. Answer the questions by using the followning tools = {{tools}}
"""
context_prompt = PromptTemplate( template=context, input_variables=[] )
#context_prompt = PromptTemplate.from_template(context)

#----------------------------------------------------------------------------------------------

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=context_prompt,
    stop_sequence=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

#----------------------------------------------------------------------------------------------
context = []
def collect_messages(event):
    user_message = inp.value.strip()
    if not user_message:
        return
    context.append({"role": "user", "content": user_message})
    try:
        agent_response = agent_executor.invoke({
            "input": user_message,
            "response": agent_response 
        })
        if agent_response:
         return agent_content
        else:
         return ("Sorry, I couldn't understand that.")
    except OutputParserException as e:
        agent_content = f"Error: {e}"
    
    context.append({"role": "assistant", "content": agent_content})
    inp.value = ""
    update_chat_display()

def update_chat_display():
    chat_contents = []  
    for msg in context:
        role = msg['role'].capitalize()
        content = msg['content']
        if role == "User":
            chat_contents.append(f'<div style="text-align: right;"><span style="background-color: #DCF8C6; padding: 8px; border-radius: 10px; display: inline-block;">You: {content}</span></div>')
        else:
            chat_contents.append(f'<div style="text-align: left;"><span style="background-color: #E6E6E6; padding: 8px; border-radius: 10px; display: inline-block;">Bot: {content}</span></div>')
    
    message_display.object = (
        "<div id='chat-container' style='height: 400px; overflow-y: auto;'>"
        + "<br>".join(chat_contents)
        + "</div>"
    )
#----------------------------------------------------------------------------------------------

message_display = pn.pane.HTML("<div style='height: 400px; overflow-y: auto;'></div>", height=450, width=800, align='center')
inp = pn.widgets.TextInput(value="", placeholder='Enter your message here...', width=300, align="center")
button_conversation = pn.widgets.Button(name="Send", button_type='primary', align="center")
button_conversation.on_click(collect_messages)
title_pane = pn.pane.Markdown("### Electronic Store Chatbot", align="center")

layout = pn.Column(
    title_pane,
    message_display,
    pn.Row(inp, button_conversation, align='center'),
    sizing_mode="stretch_width",
    align="center",
)

layout.css = """
    .bk-root {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .panel-column {
        max-width: 600px;
        width: 100%;
        margin: 0 auto;
    }
    .panel-html {
        overflow-y: scroll; 
        height: 400px;      
    }
"""

if __name__ == "__main__":
    pn.serve(layout, port=5006, title="Chatbot Interface")