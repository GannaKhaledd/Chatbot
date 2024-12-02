import panel as pn

def setup_chat_ui(agent_executor, context):
    """
    Sets up a chatbot UI using Panel with custom HTML/CSS for styling.
    Parameters:
        agent_executor: The agent executor instance for processing user inputs.
        context: A list to maintain the conversation context between the user and the bot.
    """
    pn.extension()

    message_display = pn.pane.HTML("<div style='height: 400px; overflow-y: auto;'></div>", height=450, width=800, align='center')

    # input box and send button
    inp = pn.widgets.TextInput(name="Your Message", placeholder="Type your message here...", width=300, align="center")
    send_btn = pn.widgets.Button(name="Send", button_type="primary", width=100, align="center")

    def update_chat_display():
        """Update the chat display with the conversation history."""
        chat_contents = []  
        for msg in context:
            role = msg['role'].capitalize()
            content = msg['content']
            if role == "User":
                chat_contents.append(
                    f'<div style="text-align: right;">'
                    f'<span style="background-color: #DCF8C6; padding: 8px; border-radius: 10px; display: inline-block;">'
                    f'You: {content}</span></div>'  # User's messages in a green bubble
                )
            else:
                chat_contents.append(
                    f'<div style="text-align: left;">'
                    f'<span style="background-color: #E6E6E6; padding: 8px; border-radius: 10px; display: inline-block;">'
                    f'Bot: {content}</span></div>'  # Bot's responses in a gray bubble
                )
        
        message_display.object = (
            "<div id='chat-container' style='height: 400px; overflow-y: auto;'>"
            + "<br>".join(chat_contents)
            + "</div>"
        )

    def collect_messages(event):
        """Collect user messages and display bot responses."""
        user_message = inp.value.strip()
        if not user_message:
            return  # Ignore empty input
        context.append({"role": "user", "content": user_message})
        inp.value = ""  # Clear input field
        update_chat_display()

        try:
            # Process user input with the agent executor
            agent_response = agent_executor.invoke({"input": user_message})
            if agent_response and 'output' in agent_response:
                bot_response = agent_response['output']
            else:
                bot_response = "Sorry, I couldn't process your request."

            # Append bot response to context
            context.append({"role": "assistant", "content": bot_response})
            update_chat_display()

        except Exception as e:
            context.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})
            update_chat_display()

    # Collect_messages function to the send button
    send_btn.on_click(collect_messages)

    title_pane = pn.pane.Markdown("### Electronic Store Chatbot", align="center")

    layout = pn.Column(
        title_pane,
        message_display,
        pn.Row(inp, send_btn, align='center'),
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
        #chat-container {
            padding: 10px;
        }
        .panel-input {
            margin-bottom: 10px;
        }
    """

    return layout
