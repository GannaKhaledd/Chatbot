import os  
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
from langchain.agents import initialize_agent, AgentType
from langchain_community.document_loaders import CSVLoader 
from langchain_core.exceptions import OutputParserException
from langchain_community.embeddings import SentenceTransformerEmbeddings


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
load_dotenv(find_dotenv())
api_key = os.environ.get("GROQ_API_KEY")  

client = Groq(api_key=api_key)
llm_model  = "llama3-groq-70b-8192-tool-use-preview"

persist_directory = 'doc/chroma/'
if os.path.exists(persist_directory): 
    shutil.rmtree(persist_directory) 

file = 'products.csv'
loader = CSVLoader(file_path=file, encoding='utf-8')  
data = loader.load()  

#splits each product's details into a dictionary, and then formats it into a string. 
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
def search_electronic_products(query, k=2):
    """
    I have to Search for Electronic products when the user asks and provide results.
    I Don't use this tool when the user asks about any other thing rather than the electronic products
    If there is no products avaialble I have to tell him in a friendly manner.
    Arguments:
        query (str): The search query.
        k (int): Number of top results to return. Default is 2.
    Returns:
        str: A formatted string with search results.
    """
    try:
        results = vector_store.similarity_search(query, k=k)
        if not results:
            return "No matching products found."
        
        response = ""
        counter = 1
        for result in results:
            product_details = result.page_content
            response = response + str(counter) + ". " + product_details + "\n\n"
            counter = counter + 1
        return response.strip()
    except Exception as e:
        return f"An error occurred during search: {e}"
#----------------------------------------------------------------------------------------------
cart = []
def add_to_cart(product_name):
    """
    Add products to the cart after the customer asks to.
    Arguments: 
       product_name (str) : The product that will be added to the cart by the user
    Returns:
      A confirmation message
    """
    print(f"Adding product: {product_name}")  
    for doc in data:
        content_dict = {}
        if content_dict['Product'].lower() == product_name.lower():
            cart.append(content_dict)
            print(f"Product added: {content_dict}")  
            return content_dict['Product'] + " has been added to your cart." 
    print("Product not found") 
    return "The product you are trying to add to your cart is not available right now."
#-------------------------------------------------------------------------------------------------------
def calculate_total_price(input_str=None):
    """
    Calculate the total price of items in the cart after making an order to proceed to payment process.
    This tool doesn't take any inputs and it outputs the total price of items in cart
    Arguments :
    Doesn't take any argument
    Returns :
    the total price of the products in the cart
    """
    try:
       total_price = 0
       for item in cart:
        print(item)
        total_price += float(item['Price'])
       return total_price
    except ValueError:
        return "Error: Invalid price format in cart."
   
#-------------------------------------------------------------------------------------------------------
# def make_an_order(*args):
#     """
#     I will use this tool when the user wants to make an order after adding the product to the cart
#     Arguments : 
#     productName(str) : The product the user added to the cart
#     returns:
#     The order after added to the cart. 
#     """
#     if len(cart) == 0 : 
#         return "Your cart is empty, please fill your cart and come back again."
#     order = ""
#     print(cart)
   
#     for product in cart:
#        order += product["Product"]+ ", " 
#     return f'Your order is: {order}'  

# def make_an_order():
#     """
#     I will use this tool when the user wants to make an order after adding the product to the cart
#     returns:
#     The order after added to the cart. 
#     """
#     if len(cart) == 0: 
#         return "Your cart is empty, please fill your cart and come back again."
    
#     order_details = []
#     for product in cart:
#         order_details.append(product["Product"])
    
#     total_price = calculate_total_price()
#     order_summary = f'Your order is: {", ".join(order_details)}. Total price: {total_price}'
#     return order_summary 
def make_an_order():
    """
    I will use this tool only when the user asks to make an order, after adding products to the cart.
    Don't use the tool until the user asks to make an order
    Argument:
    Doesn't take any argument
    Returns:
    A string summarizing the order.
    """
    if len(cart) == 0:
        return "Your cart is empty, please fill your cart and come back again."
    
    order_summary = "Your order contains the following products:\n"
    for product in cart:
        order_summary += f"- {product['Product']} (Price: {product['Price']})\n"
    return order_summary

#-------------------------------------------------------------------------------------------------------
def do_payment(input_str=None):
    """
    Process payment for the order and return the order details to the customer.
    Arguments :
    Doesn't take any argument
    Returns :
    the payment is done successfuly or not
    """
    if len(cart) == 0: 
        return "Your cart is empty, please fill your cart and come back again."
    
    order = make_an_order() 
    payment_method = input("How would you like to pay? (Cash/Credit card):").strip().lower()
    
    if payment_method == 'cash':
        address = input("Please provide your address for the delivery: ")
        total_price = calculate_total_price()  
        return f"Your order price is {total_price}. It will be delivered to {address}. Thank you for your order! Goodbye!"
    
    elif payment_method == 'credit card':
        credit_details = input("Please enter your credit card details: ")
        payment_successful = True 
        if payment_successful:
            total_price = calculate_total_price()  
            order_message = f"Payment was successful for {order}."
            address = input("Please provide your address for the delivery: ")
            delivery_message = f"Your order will be delivered to {address}. Thank you for your order! Goodbye!"
            return order_message + "\n" + delivery_message
        else:
            return "There was an issue with your credit card payment. Please try again."
    else:
        return "Invalid payment method selected. Please choose 'Cash' or 'Credit card'."
#-------------------------------------------------------------------------------------------------------
tools = [
     Tool(
        name="Search for Electronic Products",
        func=search_electronic_products,
        description="""
        I have to Search for Electronic products when the user asks and provide results.
        I Don't use this tool when the user asks about any other thing rather than the electronic products
        If there is no products avaialble I have to tell him in a friendly manner.
        Arguments:
        query (str): The search query.
        k (int): Number of top results to return. Default is 2.
        Returns:
        str: A formatted string with search results.
        """
    ),
    Tool(
        name="Add to Cart",
        func=add_to_cart,
        description= """
        Add products to the cart after the customer asks to.
        Arguments: 
        product_name (str) : The product that will be added to the cart by the user
        Returns:
        The dictionary of the products
        """
    ),
    Tool(
        name="Make an Order",
        func=make_an_order,
        description="""
        I will use this tool only when the user asks to make an order, after adding products to the cart.
        Don't use the tool until the user asks to make an order
        Argument:
        Doesn't take any argument
        Returns:
        A string summarizing the order.
        """
    ),
    Tool(
        name="Calculate Total Price",
        func=calculate_total_price,
        description=""" 
        Calculate the total price of items in the cart after making an order to proceed to payment process.
        This tool doesn't take any inputs and it outputs the total price of items in cart
        Arguments :
        Doesn't take any argument
        Returns :
        the total price of the products in the cart
        """
    ),
    Tool(
        name="Do Payment",
        func=do_payment,
        description="""
        Process payment for the order and return the order details to the customer.
        Arguments :
        Doesn't take any argument
        Returns :
        the payment is done successfuly or not
        """
    ),
]
tool_names = ['Search for Electronic Products','Add to Cart', 'Make an Order','Calculate Total Price', 'Do Payment']
#----------------------------------------------------------------------------------------------
prompt = """
You are a friendly bot designed to assist customers with questions about an electronic store only.
Keep the conversation centered around the electronic store only and dont go to unrelated topics
Instructions:
1. **Out of Context**: Always answer that you don't have information on any other topic rather than the electronic store  
3. **Handling Harmful Content**: If questions contain harmful or inappropriate content, politely inform the user that you cannot assist with such inquiries.
4. **Product availability**: If the customer asks for a product that is not available, handle this in a kind way
5.Answer the questions as best you can. You have access to the following tools: {tools} 

6.Please use the following format 
Thought : You should always think about what you need to do 
Action : The action is to use one of the tools 
Action Input : The input to the action (If the action is not none)
Observation : Result to the action 
Thought : I Know the final answer 
Final Answer : return the final answer

for example : 

1.User input : Hello
Thought: Do I need to use a tool? No
Action: None
Action Input: None
Observation: No action is needed this is a greeting, reply in friendly tone
Thought : I now know the final answer
Final Answer : Hello, How can I help you today !

2.User input : Can you help me with my homework
Thought: Do I need to use a tool? no
Action:  none
Action Input : None
Observation: No action is needed this is something out pf context, reply in friendly tone
Thought : I now know the final answer
Final Answer : Sorry but I can't help you with that, The only thing I can help with is the electronic products


3. User input : I want to add this laptop to the cart
Thought: Do I need to use a tool? Yes
Action: Add to Cart 
Action Input : I want to add this laptop to the cart
Observation: I will use the add to cart tool to add the laptop to the cart list
Thought : I now know the final answer
Final Answer : Your laptop is added to the cart successfully

4.User input : What is the total price of the order
Thought: Do I need to use a tool? Yes
Action:  Calculate Total Price
Action Input : None
Observation: I will use the calculate total price tool to sum the prices of products added to the cart
Thought : I now know the final answer
Final Answer : Your total price is 1000



Begin!

Question: {input}
{agent_scratchpad}
"""
context_prompt = PromptTemplate(
    template=prompt,
    input_variables=['tools','tool_names','agent_scratchpad','input']
)
llm = ChatGroq(temperature=0.0, model=llm_model)

def custom_parsing_error_handler(exception: OutputParserException) -> str:
    return f"Parsing error encountered: {str(exception)}"

#----------------------------------------------------------------------------------------------

agent = initialize_agent(
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
    tools=tools,
    memory = memory,
    verbose = True,
    handle_output_parser_exceptions=custom_parsing_error_handler 
)
agent_executor = agent
#----------------------------------------------------------------------------------------------
context=[]

# def collect_messages(event):
#     user_message = inp.value.strip()
#     if not user_message:
#         return
#     context.append({"role": "user", "content": user_message})
#     inp.value = "" 
#     update_chat_display() 
#     try:
#         agent_response = agent_executor.invoke({
#             "input": user_message
#         })
        
#         if agent_response and 'output' in agent_response:
#             bot_response = agent_response['output']
#             context.append({"role": "assistant", "content": bot_response})
#         else:
#             context.append({"role": "assistant", "content": "Sorry, I couldn't understand that."})
        
#     except OutputParserException as e:
#         context.append({"role": "assistant", "content": f"Error: {e}"})

#     update_chat_display()

def collect_messages(event):
    user_message = inp.value.strip()
    if not user_message:
        return
    context.append({"role": "user", "content": user_message})
    inp.value = "" 
    update_chat_display() 
    
    try:
        agent_response = agent_executor.invoke({"input": user_message})
        
        if agent_response and 'output' in agent_response:
            bot_response = agent_response['output']
        else:
            bot_response = "Sorry, I couldn't understand that."
        
    except OutputParserException as e:
        bot_response = custom_parsing_error_handler(e)
    except Exception as e:
        bot_response = f"An unexpected error occurred: {e}"
    
    context.append({"role": "assistant", "content": bot_response})
    update_chat_display()

#----------------------------------------------------------------------------------------------
def update_chat_display():
    chat_contents = []  
    for msg in context:
        role = msg['role'].capitalize()
        content = msg['content']
        if role == "User":
            chat_contents.append(
                f'<div style="text-align: right;">'
                f'<span style="background-color: #DCF8C6; padding: 8px; border-radius: 10px; display: inline-block;">'
                f'You: {content}</span></div>'
            )
        else:
            chat_contents.append(
                f'<div style="text-align: left;">'
                f'<span style="background-color: #E6E6E6; padding: 8px; border-radius: 10px; display: inline-block;">'
                f'Bot: {content}</span></div>'
            )
    
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