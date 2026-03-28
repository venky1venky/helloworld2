"""
Gradio Chat Interface for ACME Customer Service Training
...
"""

import os
import requests
import gradio as gr
import uuid
from pathlib import Path
from typing import List, Tuple, Any
from pydantic_ai.messages import BinaryContent
import logging

from multimodal_moderation.env import USER_API_KEY, API_BASE_URL
from multimodal_moderation.tracing import setup_tracing, get_tracer, add_media_to_span
from multimodal_moderation.agents.customer_agent import customer_agent
from multimodal_moderation.utils import detect_file_type
from opentelemetry import trace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

setup_tracing()
tracer = get_tracer(__name__)

MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

MODERATION_CONFIG = {
    "text": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_text",
        "unsafe_flags": ["is_unfriendly", "is_unprofessional", "contains_pii"],
    },
    "image": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_image_file",
        "unsafe_flags": ["contains_pii", "is_disturbing", "is_low_quality"],
    },
    "video": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_video_file",
        "unsafe_flags": ["contains_pii", "is_disturbing", "is_low_quality"],
    },
    "audio": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_audio_file",
        "unsafe_flags": ["is_unfriendly", "is_unprofessional", "contains_pii"],
    },
}


def _call_text_moderation(text: str, span: trace.Span) -> Tuple[dict[str, Any], str, str, str]:
    content_type = "text"
    mime_type = "text/plain"
    config = MODERATION_CONFIG[content_type]

    response = requests.post(
        config["endpoint"], headers={"Authorization": f"Bearer {USER_API_KEY}"}, json={"text": text}
    )

    if not response.ok:
        raise RuntimeError(f"Moderation service unavailable. Please try again later. {response.text}")

    span.set_attributes(
        {
            "input.text.content": text,
            "input.text.length": len(text),
        }
    )

    result = response.json()
    feedback = result["rationale"]

    return result, feedback, content_type, mime_type


def _call_media_moderation(media: str, span: trace.Span) -> Tuple[dict[str, Any], str, str, str]:
    mime_type = detect_file_type(media, context=media)

    file_size = os.path.getsize(media)
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise ValueError(f"File too large: {size_mb:.1f}MB. Maximum size is {MAX_FILE_SIZE_MB}MB.")

    content_type = mime_type.split("/")[0]
    config = MODERATION_CONFIG[content_type]

    with open(media, "rb") as f:
        response = requests.post(
            config["endpoint"], headers={"Authorization": f"Bearer {USER_API_KEY}"}, files={"file": f}
        )

    add_media_to_span(span, media, f"{content_type}_moderation", 0)

    if not response.ok:
        raise RuntimeError(f"Moderation service unavailable. Please try again later. {response.text}")

    result = response.json()
    feedback = result["rationale"]

    if content_type == "audio" and "transcription" in result:
        feedback = f"Transcription: \"{result['transcription']}\"\n\n{feedback}"

    return result, feedback, content_type, mime_type


def check_content_safety(*, text: str | None = None, media: str | None = None) -> Tuple[bool, str, str]:
    # TODO: use the tracer to create a span named "moderate_text"
    with tracer.start_as_current_span("moderate_text") as span:

        if text is not None:
            result, feedback, content_type, mime_type = _call_text_moderation(text, span)
        elif media is not None:
            result, feedback, content_type, mime_type = _call_media_moderation(media, span)
        else:
            raise ValueError("Must provide exactly one of text or media")

        span.set_attributes({f"output.{k}": v for k, v in result.items()})
        span.update_name(f"moderate_{content_type}")

    config = MODERATION_CONFIG[content_type]
    for flag in config["unsafe_flags"]:
        if result[flag]:
            return False, f"Content flagged: {feedback}", mime_type

    return True, feedback, mime_type


class ChatSessionWithTracing:

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        # TODO: use the tracer to create a span named "conversation"
        self.conversation_span = tracer.start_span(
            "conversation",
            attributes={"session.id": self.session_id}
        )

    async def chat_with_gemini(self, message: dict, history: List, past_messages: List) -> Tuple[str, List, str]:

        # TODO: use the tracer to create a span named "chat_turn"
        with tracer.start_as_current_span(
            "chat_turn",
            context=trace.set_span_in_context(self.conversation_span)
        ) as span:

            logger.info(f"New turn - Text: '{message.get('text', '')[:50]}...', Files: {len(message.get('files', []))}")

            prompt_parts: List[str | BinaryContent] = [
                "This is the next message from the support agent:",
            ]

            safety_message = ""

            for key, value in message.items():

                if key == "text" and value:
                    is_safe, safety_message, mime_type = check_content_safety(text=value)

                    if not is_safe:
                        feedback = f"⚠️ Content flagged: {safety_message}"
                        response = "[This content was flagged by moderation and not sent to the AI. Please try again.]"

                        # TODO: set an attribute "feedback" in the tracing span
                        with tracer.start_as_current_span("feedback") as feedback_span:
                            feedback_span.set_attribute("feedback", feedback)

                        return response, past_messages, feedback

                    prompt_parts.append(value)

                elif key == "files" and value:

                    for file_path in value:

                        try:
                            is_safe, safety_message, mime_type = check_content_safety(media=file_path)

                            if not is_safe:
                                feedback = f"⚠️ Content flagged: {safety_message}"
                                response = (
                                    "[This content was flagged by moderation and not sent to the AI. Please try again.]"
                                )
                                return response, past_messages, feedback

                            with open(file_path, "rb") as f:
                                file_bytes = f.read()

                            # TODO: create a BinaryContent object and append to prompt_parts
                            prompt_parts.append(BinaryContent(data=file_bytes, media_type=mime_type))

                        except ValueError as e:
                            raise gr.Error(str(e))

            if not prompt_parts:
                raise gr.Error("Please provide a message or at least one file.")

            try:
                with tracer.start_as_current_span("llm_customer"):
                    result = await customer_agent.run(
                        prompt_parts,
                        message_history=past_messages,
                    )

                logger.info(f"Response generated ({len(result.all_messages())} messages in history)")
                return result.output, result.all_messages(), safety_message

            except Exception as e:
                logger.error(f"Error in chat_with_gemini: {str(e)}")
                raise gr.Error(
                    f"I'm sorry, but I encountered an error while processing your request. "
                    f"Please try again or contact ACME support if the issue persists."
                )

    def end_conversation(self):
        if self.conversation_span:
            self.conversation_span.end()
            logger.info(f"Conversation {self.session_id} ended")
        return "Conversation ended. Refresh the page to start a new session."


def create_chat_interface() -> gr.Blocks:
    chat_session = ChatSessionWithTracing()

    with gr.Blocks(title="ACME Customer Service Training Agent", fill_height=True) as demo:
        past_messages_state = gr.State([])

        feedback_display = gr.Textbox(
            label="💬 Moderation Agent Feedback",
            placeholder="No feedback yet",
            interactive=False,
            visible=True,
            lines=10,
            render=False,
        )

        gr.Markdown("# 🤖 ACME Customer Service Training Agent")
        gr.Markdown("Welcome to ACME Corporation's customer service training!")

        with gr.Row():
            with gr.Column(scale=3):
                # TODO: fill the missing arguments to gr.ChatInterface
                gr.ChatInterface(
                    fn=chat_session.chat_with_gemini,
                    type="messages",
                    multimodal=True,
                    editable=False,
                    textbox=gr.MultimodalTextbox(
                        file_count="multiple",
                        file_types=["image", "video", "audio"],
                        sources=["upload", "microphone"],
                        placeholder="Type a message, upload files, or record audio...",
                    ),
                    chatbot=gr.Chatbot(
                        show_copy_button=True,
                        type="messages",
                        placeholder="👋 Start by greeting the customer or introducing yourself. The AI customer will respond with their complaint.",
                        height="75vh",
                    ),
                    additional_inputs=[past_messages_state],
                    additional_outputs=[past_messages_state, feedback_display],
                )

            with gr.Column(scale=1):
                feedback_display.render()

                end_button = gr.Button("📞 End Conversation", variant="secondary")
                end_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=False,
                )

                gr.Markdown("### 📋 Chat Guidelines")
                gr.Markdown(
                    """
                The AI acts as a customer complaining about an ACME product. Try to resolve the customer's issue.
                You can type messages, upload images/videos, or record audio.
                """
                )

                gr.Markdown("### 🔒 Content Moderation")
                gr.Markdown(
                    """
                All messages and media are automatically checked for:
                - Inappropriate content
                - Personally identifiable information
                - Unprofessional language
                """
                )

        end_button.click(fn=chat_session.end_conversation, outputs=end_status).then(
            lambda: gr.Textbox(visible=True), outputs=end_status
        )

    return demo


def main():
    demo = create_chat_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()