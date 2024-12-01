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
        try:
            lines = doc.page_content.split('\n')
            content_dict = {}
            for line in lines:
                if ": " in line:
                    key, value = line.split(": ", 1)  # Avoid splitting more than once
                    content_dict[key.strip()] = value.strip()
            if all(key in content_dict for key in ['Category', 'Product', 'Price', 'Description']):
                content = (
                    f"Category: {content_dict['Category']}\n"
                    f"Product: {content_dict['Product']}\n"
                    f"Price: {content_dict['Price']}\n"
                    f"Description: {content_dict['Description']}"
                )
                documents.append(Document(page_content=content))
        except Exception as e:
            print(f"Skipping malformed document: {e}")
    return documents

documents = create_documents(data)

embedding = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#----------------------------------------------------------------------------------------------   
def search_electronic_products(query, k=1):
    """
    Search for electronic products and return results as a list of dictionaries.
    Arguments:
    query (str): The search query for products (e.g., "iPhone").
    k (int): Number of top results to return. Default is 1.
    Returns:
    list: A list of dictionaries with product details (e.g., category, product name, price, description).
    """
    if not query or type(query) != str:
        return "Please provide a valid product name to search for."
    try:
        results = vector_store.similarity_search(query, k=k)
        if not results:
            return []

        #list hold product details
        product_details_list = []
        for result in results:
            lines = result.page_content.split("\n")
            content_dict = {}
            for line in lines:
                if ": " in line:
                    key, value = line.split(": ", 1)
                    content_dict[key.strip()] = value.strip()
            required_keys = ['Category', 'Product', 'Price', 'Description']
            #add product to list if all keys are available
            if all(key in content_dict for key in required_keys):
                product_details_list.append(content_dict)
        return product_details_list
    except Exception as e:
        return f"An unexpected error occurred during the search: {e}"
#----------------------------------------------------------------------------------------
cart = []
def add_to_cart(product_name):
    """
    Add multiple products to the cart after searching for it.
    Arguments:
        product_names (list): A list of product names to add to the cart.
    Returns:
        str: A message telling whether the product was added or not.
    """
  
    search_results = search_electronic_products(product_name, k=1)
    if isinstance(search_results, str):  # Check if it's an error message
        return search_results  # Return the error message directly
    
    if not search_results:
        return f"Sorry, I couldn't find '{product_name}'."
    
    product_details = search_results[0] 
    for item in cart:
        if item['Product'] == product_details.get('Product'):
            return f"{product_details.get('Product','Unknown Product')} is already in your cart."
    
    cart.append(product_details)  
    return f"{product_details.get('Product', 'Unknown Product')} has been added to your cart."
#-------------------------------------------------------------------------------------------------------
def calculate_total_price(input_str=None):
    """
    Calculate the total price of items in the cart when products are added.
    This function doesn't take any input and returns the total price of items in the cart.
    Arguments:
    None
    Returns:
    The total price of the products in the cart as a float or a message if there's an error.
    """
    try:
        total_price = 0
        print(f"Calculating total price for cart: {cart}") 
        for item in cart:
            if isinstance(item, dict) and 'Price' in item:
                price_str = item['Price']
                price = price_str.replace('$', '').replace(',', '')
                total_price += float(price)
            else:
                return f"Error: Unexpected item in cart: {item}"
        return total_price
    except (ValueError, KeyError) as e:
        return f"Error: Invalid price format or missing data in the cart. {str(e)}"
#-------------------------------------------------------------------------------------------------------
def make_an_order(cart):
    """
    Create an order by summarizing the products in the cart.
    This function will only be used when the user asks to make an order.
    It summarizes the products and their prices from the cart.
    Arguments:
        cart (list): List of product dictionaries in the cart.
    Returns:
        str: A string summarizing the order or informing if the cart is empty.
    """
    if len(cart) == 0:
        return "Your cart is empty, please add some products to your cart and come back again."
    
    order_summary = "Your order contains the following products:\n"
    for product in cart:
        if isinstance(product, dict) and 'Product' in product and 'Price' in product:
            order_summary += f"- {product['Product']} (Price: {product['Price']})\n"
        else:
            print(f"Unexpected item in cart: {product}")  
    
    total_price = calculate_total_price()
    if isinstance(total_price, str):
        return total_price
    
    order_summary += f"\nTotal price: ${total_price:.2f}"
    order_summary += "\n\nCould you please provide your shipping address to proceed with the order?"
    return order_summary

#-------------------------------------------------------------------------------------------------------
tools = [
     Tool(
        name="Search for Electronic Products",
        func=search_electronic_products,
        description="""
        I have to Search for Electronic products when the user asks and provide results.
        I Don't use this tool when the user asks about any other thing rather than the electronic products
        Arguments:
        query (str): The search query for products.
        k (int): Number of top results to return. Default is 1.
        Returns:
        str: A formatted string with search results or a friendly error message.
    """
    ),
    Tool(
        name="Add to Cart",
        func=add_to_cart,
        description= """
        Add a product to the cart after searching for it.
        Arguments:
        product_name (str): The name of the product to add.
        Returns:
        str: A message about the product being added or not found.
    """
    ),
    Tool(
        name="Calculate Total Price",
        func=calculate_total_price,
        description=""" 
        Calculate the total price of items in the cart when products are added.
        Don't use this tool until the 
        This function doesn't take any input and returns the total price of items in the cart.
        Arguments:
        None
        Returns:
        The total price of the products in the cart as a float or a message if there's an error.
    """
    ),
    Tool(
        name="Make an Order",
        func=make_an_order,
        description="""
        Create an order summary with a list of product names and a sentence summarizing the order.
        Arguments:
        cart (list): List of product dictionaries in the cart.
        Returns:
        tuple: A list of product names and a string summarizing the order.
    """
    ),
]
tool_names = ['Search for Electronic Products','Add to Cart','Calculate Total Price', 'Make an Order']
#----------------------------------------------------------------------------------------------
prompt = """
Instructions:
1. You are a friendly bot designed to assist customers with questions about an electronic stores and don't respond to any questions out of context.
2. **Handling Harmful Content**: If questions contain harmful or inappropriate content, politely inform the user that you cannot assist with such inquiries.
3. **Tool Usage**: 
   - Use the **Add to Cart** tool when the user requests to add an item to the cart.
   - When you use the **Make an Order** tool, after returning the string summarizing the order, you should ask the user about the address/
     of his home and then tell him in a friendly way that the order will be shipped to the address he/she provided
4.Answer the questions as best you can. You have access to the following tools: {tools} 

5..Please use the following format 
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
llm = ChatGroq(temperature=0.2, model=llm_model)
#----------------------------------------------------------------------------------------------

def custom_parsing_error_handler(exception: OutputParserException) -> str:
    return f"Parsing error encountered: {str(exception)}"
#pydantic output parser 
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

print (memory)