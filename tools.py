from langchain.agents import Tool

cart = []

# Tool 1 : Search Electronic Products
def search_electronic_products(query, vector_store, k=1):
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

        product_details_list = []
        for result in results:
            lines = result.page_content.split("\n")
            content_dict = {}
            for line in lines:
                if ": " in line:
                    key, value = line.split(": ", 1)
                    content_dict[key.strip()] = value.strip()
            required_keys = ['Category', 'Product', 'Price', 'Description']
            if all(key in content_dict for key in required_keys):
                product_details_list.append(content_dict)
        return product_details_list
    except Exception as e:
        return f"An unexpected error occurred during the search: {e}"


# Tool 2 : Add to Cart
def add_to_cart(product_name, vector_store):
    """
    Add multiple products to the cart after searching for it.
    Arguments:
        product_name (str): The name of the product to add to the cart.
    Returns:
        str: A message telling whether the product was added or not.
    """
    search_results = search_electronic_products(product_name, vector_store, k=1)
    if isinstance(search_results, str):  # Check if it's an error message
        return search_results

    if not search_results:
        return f"Sorry, I couldn't find '{product_name}'."

    product_details = search_results[0]
    for item in cart:
        if item['Product'] == product_details.get('Product'):
            return f"{product_details.get('Product', 'Unknown Product')} is already in your cart."

    cart.append(product_details)
    return f"{product_details.get('Product', 'Unknown Product')} has been added to your cart."


# Tool 3 : Calculate Total Price
def calculate_total_price(input_str=None):
    """
    Calculate the total price of items in the cart when products are added.
    Returns:
    float: Total price of the products in the cart or an error message.
    """
    try:
        total_price = 0
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


# Tool 4 : Make an Order
def make_an_order(cart):
    """
    Create an order by summarizing the products in the cart.
    Returns:
    str: A string summarizing the order or informing if the cart is empty.
    """
    if len(cart) == 0:
        return "Your cart is empty, please add some products to your cart and come back again."

    order_summary = "Your order contains the following products:\n"
    for product in cart:
        if isinstance(product, dict) and 'Product' in product and 'Price' in product:
            order_summary += f"- {product['Product']} (Price: {product['Price']})\n"

    total_price = calculate_total_price()
    if isinstance(total_price, str):
        return total_price

    order_summary += f"\nTotal price: ${total_price:.2f}"
    order_summary += "\n\nCould you please provide your shipping address to proceed with the order?"
    return order_summary


# Tool Definitions
def get_tools(vector_store):
    return [
        Tool(
            name="Search for Electronic Products",
            func=lambda query: search_electronic_products(query, vector_store),
            description="""Search for electronic products and return results as a list of dictionaries."""
        ),
        Tool(
            name="Add to Cart",
            func=lambda product_name: add_to_cart(product_name, vector_store),
            description="""Add a product to the cart after searching for it."""
        ),
        Tool(
            name="Calculate Total Price",
            func=calculate_total_price,
            description="""Calculate the total price of items in the cart."""
        ),
        Tool(
            name="Make an Order",
            func=make_an_order,
            description="""Create an order summary with a list of product names and total price."""
        ),
    ]

