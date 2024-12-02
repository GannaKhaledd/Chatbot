from langchain.schema import Document

def create_documents(data):
    documents = []
    for doc in data:
        try:
            lines = doc.page_content.split('\n')
            content_dict = {}
            for line in lines:
                if ": " in line:
                    key, value = line.split(": ", 1)
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