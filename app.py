import re

from pathlib import Path

import streamlit as st
import yaml

from openai import OpenAI


st.title("Metaprompt Generator")

st.markdown("""
The metaprompt framework was originally developed by Anthropic's team to help users create clear, structured instructions for their AI assistant, Claude.
This framework supports accurate and consistent task performance by providing practical principles and examples for prompt engineering.
You can find more information and examples in Anthropic's documentation here:
[Anthropic's Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-generator).
""")

api_key = (
    st.text_input(
        "Enter your OpenAI API key",
        type="password",
        help="Enter your API key to proceed.",
    )
    or st.secrets["OPENAI_API_KEY"]
)

if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.error("Please enter your OpenAI API key.")
    st.stop()

metaprompt = Path("./metaprompt-short.txt").read_text()


@st.cache_data
def list_models():
    return [model.id for model in client.models.list()]


available_models = list_models()
MODEL = st.selectbox(
    "Select a model", options=available_models, index=available_models.index("gpt-4o")
)

st.markdown("""
Welcome to the Metaprompt! This is a prompt engineering tool designed to solve the "blank page problem" and give you a starting point for iteration. All you need to do is enter your task, and optionally the names of the variables you'd like the assistant to use in the template. Then you'll be able to run the prompt that comes out on any examples you like.

**Caveats**
- This is designed for single-turn question/response prompts, not multiturn.
- The Metaprompt is designed for use with large language models. Generating prompts with other models may lead to worse results.
- The prompt you'll get at the end is not guaranteed to be optimal by any means, so don't be afraid to change it!
""")

TASK = st.text_input(
    "Enter your task", "Draft an email responding to a customer complaint"
)

VARIABLES_INPUT = st.text_input(
    "Optional: specify the input variables you want the assistant to use (comma-separated)",
    "CUSTOMER_NAME, CUSTOMER_COMPLAINT",
)

VARIABLES = (
    [var.strip().upper() for var in VARIABLES_INPUT.split(",")]
    if VARIABLES_INPUT
    else []
)

variable_string = ""
for variable in VARIABLES:
    variable_string += f"\n{{${variable.upper()}}}"

prompt = metaprompt.replace("{{TASK}}", TASK)

assistant_partial = "<Inputs>"
if variable_string:
    assistant_partial += variable_string + "\n</Inputs>\n<Instructions Structure>"
else:
    assistant_partial += "\n</Inputs>\n<Instructions Structure>"


def generate_prompt_template(prompt, assistant_partial):
    full_prompt = prompt + "\n" + assistant_partial

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2,
    )

    completion = response.choices[0].message.content
    return completion


def extract_between_tags(tag: str, string: str, strip: bool = False) -> list[str]:
    # Ensure we escape the tag to avoid conflicts with special characters in regex
    tag_pattern = re.compile(f"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)
    ext_list = tag_pattern.findall(string)

    if strip:
        ext_list = [e.strip() for e in ext_list]

    return ext_list


def remove_empty_tags(text):
    return re.sub(r"\n<(\w+)>\s*</\1>\n", "", text, flags=re.DOTALL)


def remove_unclosed_tags(text: str) -> str:
    tags = re.findall(r"<(/?\w+)>", text)
    stack = []
    closed_text = text

    for tag in tags:
        if not tag.startswith("/"):
            stack.append(tag)
        else:
            if stack and stack[-1] == tag[1:]:
                stack.pop()
            else:
                closed_text = re.sub(f"<{tag}>", "", closed_text)

    for tag in stack:
        closed_text = re.sub(f"<{tag}>", "", closed_text)

    return closed_text


def extract_prompt(metaprompt_response):
    between_tags = extract_between_tags("Instructions", metaprompt_response)[0]
    between_tags = remove_empty_tags(between_tags).strip()
    between_tags = remove_unclosed_tags(between_tags).strip()
    return between_tags


def extract_variables(prompt):
    pattern = r"{([^}]+)}"
    variables = re.findall(pattern, prompt)
    return set(variables)


def generate_assistant_output(prompt_with_variables):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt_with_variables}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def convert_prompt_to_yaml(prompt_text):
    yaml_content = {"prompt": prompt_text}
    yaml_output = yaml.dump(yaml_content, sort_keys=False, default_flow_style=False)
    return yaml_output


if st.button("Generate Prompt Template"):
    with st.spinner("Generating prompt template..."):
        generated_text = generate_prompt_template(prompt, assistant_partial)

        st.session_state["generated_text"] = generated_text

        # If <Instructions is not closed, assume this section is not terminated.
        INSTRUCTIONS_END = "</Instructions>"
        if generated_text and INSTRUCTIONS_END not in generated_text:
            generated_text += INSTRUCTIONS_END

        with st.expander("Response"):
            st.code(generated_text, language="markdown")

        extracted_prompt_template = extract_prompt(generated_text)
        variables_in_prompt = extract_variables(extracted_prompt_template)

        st.session_state["extracted_prompt_template"] = extracted_prompt_template
        st.session_state["variables_in_prompt"] = variables_in_prompt

if "generated_text" in st.session_state:
    generated_text = st.session_state["generated_text"]
    extracted_prompt_template = st.session_state["extracted_prompt_template"]
    variables_in_prompt = st.session_state["variables_in_prompt"]

    st.subheader("Generated Prompt Template:")
    st.code(generated_text, language="markdown")

    st.subheader("Extracted Variables:")
    st.write(variables_in_prompt or None)
    st.subheader("Final Prompt Template:")
    st.code(extracted_prompt_template, language="markdown")

    with st.expander("YAML"):
        st.code(convert_prompt_to_yaml(extracted_prompt_template), language="yaml")

    st.header("Test Your Prompt Template")
    variable_values = {}
    for variable in variables_in_prompt:
        value = st.text_input(
            f"Enter value for variable {variable}", key=f"variable_value_{variable}"
        )
        variable_values[variable] = value

    prompt_with_variables = extracted_prompt_template
    for variable in variable_values:
        prompt_with_variables = prompt_with_variables.replace(
            "{" + variable + "}", variable_values[variable]
        )

    if st.button("Generate Assistant's Output"):
        with st.spinner("Generating assistant's output..."):
            assistant_output = generate_assistant_output(prompt_with_variables)
            st.subheader("Assistant's Output:")
            st.write(assistant_output)
