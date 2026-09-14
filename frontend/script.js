const questionInput = document.getElementById("question-input");
const sendButton = document.getElementById("send-button");
const newChatButton = document.getElementById("new-chat-button");
const chatBox = document.getElementById("chat-box");

// --------------------------------------------------
// Send question
// --------------------------------------------------

async function sendQuestion(questionFromButton = null) {

    const question = questionFromButton || questionInput.value.trim();

    if (!question) {
        return;
    }


    // --------------------------------------------------
    // Display employee's question
    // --------------------------------------------------

    const userMessage = document.createElement("div");

    userMessage.className = "message user-message";

    const userText = document.createElement("p");

    userText.textContent = question;

    userMessage.appendChild(userText);

    chatBox.appendChild(userMessage);


    // Clear input
    questionInput.value = "";


    // --------------------------------------------------
    // Create assistant message
    // --------------------------------------------------

    const assistantMessage = document.createElement("div");

    assistantMessage.className = "message assistant-message";

    const assistantText = document.createElement("div");

    assistantText.className = "assistant-text";

    assistantMessage.appendChild(assistantText);

    chatBox.appendChild(assistantMessage);


    try {

        const response = await fetch(
            `/ask?question=${encodeURIComponent(question)}`
        );


        if (!response.ok) {
            throw new Error("Server error");
        }


        const reader = response.body.getReader();

        const decoder = new TextDecoder();

        let answerText = "";

        let followupText = "";

        const marker = "__FOLLOWUPS__";


        // --------------------------------------------------
        // Read streamed response
        // --------------------------------------------------

        while (true) {

            const { value, done } = await reader.read();

            if (done) {
                break;
            }


            const chunk = decoder.decode(value);

            answerText += chunk;


            // --------------------------------------------------
            // Check whether follow-up marker has arrived
            // --------------------------------------------------

            const markerIndex = answerText.indexOf(marker);


            if (markerIndex !== -1) {

                // Everything before the marker is the answer
                const cleanAnswer =
                    answerText.substring(0, markerIndex);


                // Everything after the marker is follow-up JSON
                followupText =
                    answerText.substring(
                        markerIndex + marker.length
                    );


                assistantText.innerHTML =
                    formatResponse(cleanAnswer);

            } else {

                // No follow-ups yet — display normal answer
                assistantText.innerHTML =
                    formatResponse(answerText);

            }


            chatBox.scrollTop =
                chatBox.scrollHeight;
        }


        // --------------------------------------------------
        // Final cleanup after streaming is complete
        // --------------------------------------------------

        const finalMarkerIndex =
            answerText.indexOf(marker);


        if (finalMarkerIndex !== -1) {

            const cleanAnswer =
                answerText.substring(
                    0,
                    finalMarkerIndex
                );


            followupText =
                answerText.substring(
                    finalMarkerIndex + marker.length
                );


            assistantText.innerHTML =
                formatResponse(cleanAnswer);
        }


        // --------------------------------------------------
        // Display clickable follow-ups
        // --------------------------------------------------

        if (followupText.trim()) {

            displayFollowups(
                followupText
            );
        }


    } catch (error) {

        assistantText.textContent =
            "Sorry, something went wrong. Please try again.";

        console.error(error);
    }
}


// --------------------------------------------------
// Format chatbot response
// --------------------------------------------------

function formatResponse(text) {

    let formatted = text;


    // Escape HTML characters
    formatted = formatted
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");


    // Bold text
    formatted = formatted.replace(
        /\*\*(.*?)\*\*/g,
        "<strong>$1</strong>"
    );


    // Numbered lists
    formatted = formatted.replace(
        /^\s*(\d+)\.\s+(.*)$/gm,
        "<div class=\"list-item\"><span>$1.</span> $2</div>"
    );


    // Bullet lists
    formatted = formatted.replace(
        /^\s*[-*]\s+(.*)$/gm,
        "<div class=\"bullet-item\">• $1</div>"
    );


    // Paragraph breaks
    formatted = formatted.replace(
        /\n\n/g,
        "<div class=\"paragraph-break\"></div>"
    );


    // Single line breaks
    formatted = formatted.replace(
        /\n/g,
        "<br>"
    );


    return formatted;
}


// --------------------------------------------------
// Display follow-up suggestions
// --------------------------------------------------

function displayFollowups(text) {

    try {

        const cleanedText = text.trim();


        const suggestions =
            JSON.parse(cleanedText);


        if (!Array.isArray(suggestions)) {
            return;
        }


        // --------------------------------------------------
        // Create follow-up container
        // --------------------------------------------------

        const followupContainer =
            document.createElement("div");

        followupContainer.className =
            "followup-container";


        // --------------------------------------------------
        // Heading
        // --------------------------------------------------

        const heading =
            document.createElement("div");

        heading.className =
            "followup-heading";

        heading.textContent =
            "You may also want to explore:";


        followupContainer.appendChild(
            heading
        );


        // --------------------------------------------------
        // Create buttons
        // --------------------------------------------------

        suggestions.forEach(
            function(suggestion) {

                const button =
                    document.createElement("button");


                button.className =
                    "followup-button";


                button.textContent =
                    suggestion;


                button.type =
                    "button";


                button.addEventListener(
                    "click",
                    function() {

                        sendQuestion(
                            suggestion
                        );

                    }
                );


                followupContainer.appendChild(
                    button
                );

            }
        );


        // --------------------------------------------------
        // Add to chat
        // --------------------------------------------------

        chatBox.appendChild(
            followupContainer
        );


        chatBox.scrollTop =
            chatBox.scrollHeight;


    } catch (error) {

        console.error(
            "Could not parse follow-up suggestions:",
            error
        );

    }
}


// --------------------------------------------------
// Send button
// --------------------------------------------------

sendButton.addEventListener(
    "click",
    function() {

        sendQuestion();

    }
);


// --------------------------------------------------
// Enter key
// --------------------------------------------------

questionInput.addEventListener(
    "keydown",
    function(event) {

        if (event.key === "Enter") {

            sendQuestion();

        }

    }
);

// --------------------------------------------------
// New Chat
// --------------------------------------------------

newChatButton.addEventListener(
    "click",
    async function() {

        try {

            const response = await fetch("/new-chat");

            if (!response.ok) {
                throw new Error("Could not start a new chat.");
            }

            // Clear the chat window
            chatBox.innerHTML = `
                <div class="message assistant-message">
                    <p>
                        Hello! 👋 Welcome to Nexora.<br>
                        I'm the Nexora Training Assistant, your AI onboarding and training companion.
                        I can help you with company information, role-specific guidance, employee policies, and onboarding processes.<br><br>
                        How can I help you today?
                    </p>
                </div>
            `;

            // Clear the input box
            questionInput.value = "";

            // Put the cursor back in the input
            questionInput.focus();

        } catch (error) {

            console.error(
                "Could not start a new chat:",
                error
            );

        }
    }
);